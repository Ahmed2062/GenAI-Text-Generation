import requests
from bs4 import BeautifulSoup
import json
import time

poems = []

num_pages = 5

for i in range(num_pages):
    try:
        print(f"Processing page {i + 1}/{num_pages}...")
        url = f"https://poets.org/poems?page={i}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "lxml")

        # Find all poem links on the page
        poem_links = soup.find_all('td', class_="views-field views-field-title")

        for each in poem_links:
            try:
                link = each.find('a')
                if not link or not link['href']:
                    continue

                poem_url = "https://poets.org" + link['href']
                poem_page = requests.get(poem_url)
                poem_soup = BeautifulSoup(poem_page.text, 'lxml')

                # Extract title
                title_tag = poem_soup.find('h1')
                if title_tag:
                    span_tag = title_tag.find('span', class_='field field--title')
                    title = span_tag.text.strip() if span_tag else "Unknown Title"
                else:
                    title = "Unknown Title"

                # Extract body
                body = []
                for para in poem_soup.find_all('p'):
                    lines = para.find_all('span', class_='long-line')
                    if lines:
                        stanza = ' '.join(line.text.strip() for line in lines)
                        body.append(stanza)

                full_text = '\n'.join(body)
                if full_text.strip():  # only save non-empty poems
                    poems.append({"title": title, "body": full_text})

            except Exception as e:
                print(f"Error processing poem: {e}")
                continue

        # Save progress after each page
        with open('poems.json', 'w') as f:
            json.dump(poems, f, indent=2)

        time.sleep(1)

    except Exception as e:
        print(f"Error processing page {i}: {e}")
        continue

print(f"\n✅ Done! Total poems scraped: {len(poems)}")
print("Saved to poems.json")
