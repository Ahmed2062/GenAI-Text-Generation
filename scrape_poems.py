import requests
from bs4 import BeautifulSoup
import json
import time
import argparse # 1. Import the library
import os

def scrape_poems(num_pages: int, output_file: str):
    """Scrapes a specified number of pages of poems from poets.org."""
    poems = []
    print(f"Starting to scrape {num_pages} pages...")

    for i in range(num_pages):
        try:
            print(f"Processing page {i + 1}/{num_pages}...")
            url = f"https://poets.org/poems?page={i}"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "lxml")

            poem_links = soup.find_all('td', class_="views-field views-field-title")

            for each in poem_links:
                try:
                    link = each.find('a')
                    if not link or not link['href']:
                        continue

                    poem_url = "https://poets.org" + link['href']
                    poem_page = requests.get(poem_url)
                    poem_soup = BeautifulSoup(poem_page.text, 'lxml')

                    title_tag = poem_soup.find('h1')
                    if title_tag:
                        span_tag = title_tag.find('span', class_='field field--title')
                        title = span_tag.text.strip() if span_tag else "Unknown Title"
                    else:
                        title = "Unknown Title"

                    body = []
                    for para in poem_soup.find_all('p'):
                        lines = para.find_all('span', class_='long-line')
                        if lines:
                            stanza = ' '.join(line.text.strip() for line in lines)
                            body.append(stanza)

                    full_text = '\n'.join(body)
                    if full_text.strip():
                        poems.append({"title": title, "body": full_text})

                except Exception as e:
                    print(f"Error processing poem: {e}")
                    continue

            time.sleep(1)

        except Exception as e:
            print(f"Error processing page {i}: {e}")
            continue

    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(poems, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Total poems scraped: {len(poems)}")
    print(f"Saved to {output_file}")

# 2. Add this block to handle command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape poems from poets.org.")
    
    # Define the --pages argument
    parser.add_argument("--pages", type=int, default=5, help="Number of pages to scrape (default: 5)")
    
    # Define an optional --output argument for the filename
    parser.add_argument("--output", type=str, default="data/poems.json", help="Output JSON file path")

    args = parser.parse_args()

     os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 3. Call the function with the arguments provided by the user
    scrape_poems(args.pages, args.output)
