import os
import json
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/chunks"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
MIN_TOKENS = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Token counter using tiktoken (OpenAI-style tokens, robust & fast)
tokenizer = tiktoken.get_encoding("cl100k_base")


def token_length(text: str) -> int:
    return len(tokenizer.encode(text))


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=token_length,
    separators=["\n\n", "\n", " ", ""],
)


def process_documents():
    chunk_count = 0

    for file_name in os.listdir(INPUT_DIR):
        if not file_name.endswith(".json"):
            continue

        with open(os.path.join(INPUT_DIR, file_name), "r", encoding="utf-8") as f:
            doc = json.load(f)

        chunks = text_splitter.split_text(doc["text"])

        for idx, chunk_text in enumerate(chunks):
            num_tokens = token_length(chunk_text)

            if num_tokens < MIN_TOKENS:
                continue

            chunk_record = {
                "chunk_id": f"{doc['doc_id']}_chunk_{idx}",
                "doc_id": doc["doc_id"],
                "url": doc["url"],
                "title": doc["title"],
                "text": chunk_text,
                "num_tokens": num_tokens
            }

            output_path = os.path.join(
                OUTPUT_DIR,
                f"chunk_{chunk_count}.json"
            )

            with open(output_path, "w", encoding="utf-8") as out:
                json.dump(chunk_record, out, ensure_ascii=False, indent=2)

            chunk_count += 1

    print(f"Total chunks created: {chunk_count}")


if __name__ == "__main__":
    process_documents()
