import json
import os
import re

RAW_DIR = "data/raw"
CLEAN_DIR = "data/processed"

os.makedirs(CLEAN_DIR, exist_ok=True)


def clean_text(text):
    text = re.sub(r"\[\d+\]", "", text)   # remove citations
    text = re.sub(r"\s+", " ", text)
    return text.strip()


for file_name in os.listdir(RAW_DIR):
    if not file_name.endswith(".json"):
        continue

    with open(os.path.join(RAW_DIR, file_name), "r", encoding="utf-8") as f:
        doc = json.load(f)

    doc["text"] = clean_text(doc["text"])

    with open(os.path.join(CLEAN_DIR, file_name), "w", encoding="utf-8") as out:
        json.dump(doc, out, ensure_ascii=False, indent=2)

    print(f"[CLEANED] {file_name}")
