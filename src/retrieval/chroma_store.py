import os
import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_DIR = "data/processed"
DB_DIR = "chroma_db"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


def build_chroma_index():

    print("Building ChromaDB index...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=DB_DIR)

    collection = client.get_or_create_collection(
        name="rag_chunks",
        embedding_function=embed_fn
    )

    doc_id = 0

    for file in os.listdir(INPUT_DIR):

        if not file.endswith(".json"):
            continue

        with open(os.path.join(INPUT_DIR, file), "r", encoding="utf-8") as f:
            doc = json.load(f)

        chunks = splitter.split_text(doc["text"])

        for chunk in chunks:

            collection.add(
                ids=[f"{doc_id}"],
                documents=[chunk],
                metadatas=[{
                    "url": doc["url"],
                    "title": doc["title"]
                }]
            )

            doc_id += 1

    print("Chroma index built successfully!")


def chroma_query(query, top_k=5):

    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=DB_DIR)

    collection = client.get_collection(
        name="rag_chunks",
        embedding_function=embed_fn
    )

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    output = []

    for i in range(len(results["documents"][0])):

        output.append({
            "rank": i + 1,
            "text": results["documents"][0][i],
            "url": results["metadatas"][0][i]["url"],
            "title": results["metadatas"][0][i]["title"]
        })

    return output


if __name__ == "__main__":

    build_chroma_index()

    query = "How bengal was important for India in terms of Computer Science?"

    results = chroma_query(query)

    for r in results:
        print(f"[Rank {r['rank']}] {r['url']}")
