import json
import os
import re
import time
import requests
from urllib.parse import urlparse

# ================= CONFIG =================
OUTPUT_DIR = "data/raw"
MIN_WORDS = 200

FIXED_URLS_FILE = "data/fixed_urls.json"
RANDOM_URLS_FILE = "data/random_urls.json"

REQUEST_DELAY = 1.0   # seconds between requests
MAX_RETRIES = 3

API_URL = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "Hybrid-RAG-BITS/1.0 (contact: 2024aa05811@wilp.bits-pilani.ac.in)",
    "Accept-Language": "en-US,en;q=0.9"
}
# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_title_from_url(url):
    return urlparse(url).path.replace("/wiki/", "").replace("_", " ")


def clean_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ✅ Wikipedia API based fetching (NO 403 errors)
def fetch_wikipedia_page(url):
    title = extract_title_from_url(url)

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "titles": title
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(API_URL, params=params, headers=HEADERS, timeout=10)
            response.raise_for_status()

            data = response.json()
            page = next(iter(data["query"]["pages"].values()))
            text = page.get("extract", "")

            return clean_text(text)

        except Exception as e:
            print(f"[RETRY {attempt+1}] {title} -> {e}")
            time.sleep(2)

    return ""


def load_urls():
    urls = []

    if os.path.exists(FIXED_URLS_FILE):
        with open(FIXED_URLS_FILE, "r", encoding="utf-8") as f:
            urls.extend(json.load(f))

    if os.path.exists(RANDOM_URLS_FILE):
        with open(RANDOM_URLS_FILE, "r", encoding="utf-8") as f:
            urls.extend(json.load(f))

    # Remove duplicates while preserving order
    return list(dict.fromkeys(urls))


def process_urls():
    urls = load_urls()
    print(f"\n📌 Total URLs to process: {len(urls)}\n")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for idx, url in enumerate(urls):
        print(f"[INFO] Processing {idx+1}/{len(urls)}")

        text = fetch_wikipedia_page(url)
        word_count = len(text.split())

        if word_count < MIN_WORDS:
            print(f"[SKIP] Too short ({word_count} words): {url}")
            skip_count += 1
            continue

        record = {
            "doc_id": idx,
            "url": url,
            "title": extract_title_from_url(url),
            "word_count": word_count,
            "text": text
        }

        output_path = os.path.join(OUTPUT_DIR, f"doc_{idx}.json")
        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(record, out, ensure_ascii=False, indent=2)

        print(f"[OK] Saved ({word_count} words)")
        success_count += 1

        time.sleep(REQUEST_DELAY)

    print("\n========== SUMMARY ==========")
    print(f"Total URLs     : {len(urls)}")
    print(f"Saved docs     : {success_count}")
    print(f"Skipped docs   : {skip_count}")
    print(f"Failed docs    : {fail_count}")
    print("=============================")


if __name__ == "__main__":
    process_urls()
