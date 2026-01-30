import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_DIR = "data/chunks"
MODEL_NAME = "all-MiniLM-L6-v2"


class DenseRetriever:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.embeddings = []
        self.chunk_metadata = []
        self.index = None

    def load_chunks(self):
        """Load chunk text and metadata"""
        texts = []

        for file_name in os.listdir(CHUNKS_DIR):
            if not file_name.endswith(".json"):
                continue

            with open(os.path.join(CHUNKS_DIR, file_name), "r", encoding="utf-8") as f:
                chunk = json.load(f)

            texts.append(chunk["text"])
            self.chunk_metadata.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "url": chunk["url"],
                "title": chunk["title"],
                "text": chunk["text"]
            })

        print(f"Loaded {len(texts)} chunks for dense retrieval")
        return texts

    def build_index(self):
        """Embed chunks and build FAISS index"""
        texts = self.load_chunks()

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        self.embeddings = np.array(embeddings).astype("float32")
        dim = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)  # cosine similarity via inner product
        self.index.add(self.embeddings)

        print("FAISS dense index built")

    def retrieve(self, query, top_k=10):
        """Retrieve top-k chunks for a query"""
        if self.index is None:
            raise ValueError("FAISS index not built")

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for rank, idx in enumerate(indices[0], start=1):
            meta = self.chunk_metadata[idx]
            results.append({
                "rank": rank,
                "score": float(scores[0][rank - 1]),
                "chunk_id": meta["chunk_id"],
                "doc_id": meta["doc_id"],
                "url": meta["url"],
                "title": meta["title"],
                "text": meta["text"]
            })

        return results


if __name__ == "__main__":
    retriever = DenseRetriever()
    retriever.build_index()

    query = "How bengal was important for India in terms of Computer Science?"
    results = retriever.retrieve(query, top_k=5)

    for r in results:
        print(f"[Rank {r['rank']}] {r['url']} | Score: {r['score']:.4f}")
