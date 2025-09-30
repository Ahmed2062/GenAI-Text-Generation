import json
import re
import os
import argparse

def clean_poems(input_file: str, output_file: str):
    if not os.path.isfile(input_file):
        print(f"Error: File '{input_file}' does not exist.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: JSON file must contain a list.")
        return

    unique_entries = set()
    cleansed_data = []
    num_duplicates = 0

    for entry in data:
        if not isinstance(entry, dict):
            continue

        title = entry.get("title", "Untitled").strip()
        body = entry.get("body", "").strip()

        cleaned_body = re.sub(r'\n\s*\n+', '\n', body).strip()
        combined_entry = {"title": title, "poem": cleaned_body}

        key = f"{title}-{cleaned_body}"
        if key not in unique_entries:
            cleansed_data.append(combined_entry)
            unique_entries.add(key)
        else:
            num_duplicates += 1

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleansed_data, f, indent=2, ensure_ascii=False)

    print(f"Original entries: {len(data)}")
    print(f"Cleaned entries: {len(cleansed_data)}")
    print(f"Duplicates removed: {num_duplicates}")
    print(f"Cleansed data saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/poems.json", help="Input JSON file")
    parser.add_argument("--output", type=str, default="data/cleaned.json", help="Output cleaned JSON file")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    clean_poems(args.input, args.output)
