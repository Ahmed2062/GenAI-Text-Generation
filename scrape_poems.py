import requests
from bs4 import BeautifulSoup
import json
import time
import argparse
import os

# Use a User-Agent to mimic a browser, reducing the chance of being blocked.
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def scrape_poems(num_pages: int, output_file: str):
    poems = []
    print(f"Starting to scrape {num_pages} pages from poets.org...")

    for i in range(num_pages):
        try:
            print(f"Processing page {i + 1}/{num_pages}...")
            # The page query is 0-indexed on the site
            url = f"https://poets.org/poems?page={i}"
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()  # Will raise an exception for bad status codes (4xx or 5xx)
            soup = BeautifulSoup(response.text, "lxml")

            # Find all links to poems on the page
            poem_links = soup.select('td.views-field-title a')
            
            for link_tag in poem_links:
                try:
                    if not link_tag or not link_tag.get('href'):
                        continue

                    poem_url = "https://poets.org" + link_tag['href']
                    poem_page = requests.get(poem_url, headers=HEADERS)
                    poem_page.raise_for_status()
                    poem_soup = BeautifulSoup(poem_page.text, 'lxml')

                    # More robust selectors: find the main content area first
                    title = poem_soup.select_one('h1.display-5').get_text(strip=True) if poem_soup.select_one('h1.display-5') else "Unknown Title"
                    
                    # Assuming the poem is within a specific container to avoid grabbing other text
                    poem_container = poem_soup.select_one('div.text-formatted')
                    if not poem_container:
                        continue
                    
                    # Extract text, preserving line breaks within the poem structure
                    lines = [line.get_text(strip=True) for line in poem_container.find_all('div', class_='line')]
                    full_text = '\n'.join(lines)

                    if full_text:
                        poems.append({"title": title, "body": full_text})
                        print(f"  Scraped: {title}")

                except requests.exceptions.RequestException as e:
                    print(f"  --> Could not fetch poem URL: {e}")
                except Exception as e:
                    print(f"  --> Error processing a poem link: {e}")
                
                time.sleep(0.5) # Small delay between individual poem requests

            time.sleep(1) # Be polite and wait between scraping pages

        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {i + 1}: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred on page {i + 1}: {e}")
            continue

    # --- EFFICIENT SAVING ---
    # Save the entire list to the file only ONCE at the end.
    print("\nScraping complete. Saving data to file...")
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(poems, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Total poems scraped: {len(poems)}")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape poems from poets.org")
    parser.add_argument("--pages", type=int, default=5, help="Number of pages to scrape")
    parser.add_argument("--output", type=str, default="data/poems.json", help="Output JSON file path")
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    scrape_poems(args.pages, args.output)
