import argparse
import json
import time
import requests

HEADERS = {
    "User-Agent": "Hybrid-RAG-BITS/1.0 (contact: 2024aa05811@wilp.bits-pilani.ac.in)"
}

API_URL = "https://en.wikipedia.org/w/api.php"
MIN_WORDS = 200
REQUEST_DELAY = 0.6
MAX_DEPTH = 3   # ✅ change depth here


def get_category_members(category, cmtype="page", limit=50, cmcontinue=None):
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": cmtype,  # "page" or "subcat"
        "cmlimit": limit
    }

    if cmcontinue:
        params["cmcontinue"] = cmcontinue

    response = requests.get(API_URL, params=params, headers=HEADERS, timeout=10)
    response.raise_for_status()
    data = response.json()

    members = [item["title"] for item in data["query"]["categorymembers"]]
    next_token = data.get("continue", {}).get("cmcontinue")

    return members, next_token


def get_page_word_count(title):
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "titles": title
    }

    response = requests.get(API_URL, params=params, headers=HEADERS, timeout=10)
    response.raise_for_status()
    data = response.json()

    page = next(iter(data["query"]["pages"].values()))
    text = page.get("extract", "")
    return len(text.split())


def crawl_category(category, depth, urls, visited_categories, seen_pages, target_count):
    if depth > MAX_DEPTH or len(urls) >= target_count:
        return

    if category in visited_categories:
        return

    visited_categories.add(category)
    print(f"\n[CRAWL] Depth={depth} | Category: {category}")

    # ✅ 1) Collect pages in this category
    cmcontinue = None
    while True:
        pages, cmcontinue = get_category_members(category, cmtype="page", cmcontinue=cmcontinue)

        for title in pages:
            if title in seen_pages:
                continue
            seen_pages.add(title)

            try:
                wc = get_page_word_count(title)

                if wc >= MIN_WORDS:
                    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                    urls.append(url)
                    print(f"[OK] {title} ({wc} words)")

                    if len(urls) >= target_count:
                        return
                else:
                    print(f"[SKIP] {title} ({wc} words)")

                time.sleep(REQUEST_DELAY)

            except Exception as e:
                print(f"[ERROR] {title} -> {e}")
                time.sleep(1)

        if not cmcontinue or len(urls) >= target_count:
            break

    # ✅ 2) Recursively crawl subcategories
    subcat_continue = None
    while True:
        subcats, subcat_continue = get_category_members(category, cmtype="subcat", cmcontinue=subcat_continue)

        for subcat in subcats:
            subcat_name = subcat.replace("Category:", "")
            crawl_category(subcat_name, depth + 1, urls, visited_categories, seen_pages, target_count)

            if len(urls) >= target_count:
                return

        if not subcat_continue:
            break


def generate_urls(root_category, count):
    urls = []
    visited_categories = set()
    seen_pages = set()

    crawl_category(root_category, 0, urls, visited_categories, seen_pages, count)

    return urls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True, help="Root Wikipedia category")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=str, default="data/category_urls.json")
    args = parser.parse_args()

    urls = generate_urls(args.category, args.count)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(urls, f, indent=2)

    print(f"\n✅ Generated {len(urls)} URLs")
    print(f"📁 Saved to: {args.output}")


if __name__ == "__main__":
    main()
