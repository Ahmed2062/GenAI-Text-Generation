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
            url = f"https://poets.org/poems?page={i}"
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")

            # --- CHANGE #1: Find all links to poems on the page ---
            # The site now uses <h3> tags with a 'card-title' class for poem links.
            poem_links = soup.select('h3.card-title a')
            
            if not poem_links:
                print("  --> No poem links found on this page. The site structure may have changed again.")
                continue

            for link_tag in poem_links:
                try:
                    if not link_tag or not link_tag.get('href'):
                        continue

                    poem_url = "https://poets.org" + link_tag['href']
                    poem_page = requests.get(poem_url, headers=HEADERS)
                    poem_page.raise_for_status()
                    poem_soup = BeautifulSoup(poem_page.text, 'lxml')

                    # --- CHANGE #2: Find the title and poem body ---
                    # The title is now in a simple <h1> tag.
                    title = poem_soup.select_one('h1').get_text(strip=True) if poem_soup.select_one('h1') else "Unknown Title"
                    
                    # The poem body is inside a div with a specific data-testid attribute.
                    poem_container = poem_soup.select_one('[data-testid="poem__body"]')
                    if not poem_container:
                        print(f"  --> Could not find poem body for: {title}")
                        continue
                    
                    # --- CHANGE #3: Extract the full text robustly ---
                    # This is more reliable than looking for individual lines.
                    full_text = poem_container.get_text(separator='\n', strip=True)

                    if full_text:
                        poems.append({"title": title, "body": full_text})
                        print(f"  Scraped: {title}")

                except requests.exceptions.RequestException as e:
                    print(f"  --> Could not fetch poem URL: {e}")
                except Exception as e:
                    print(f"  --> Error processing a poem link for {link_tag.get('href', 'N/A')}: {e}")
                
                time.sleep(0.5)

            time.sleep(1)

        except Exception as e:
            print(f"An unexpected error occurred on page {i + 1}: {e}")
            continue

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

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    scrape_poems(args.pages, args.output)
