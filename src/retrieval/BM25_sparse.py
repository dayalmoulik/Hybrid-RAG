import os
import json
from rank_bm25 import BM25Okapi

CHUNKS_DIR = "data/chunks"


class BM25Retriever:
    def __init__(self):
        self.corpus = []
        self.chunk_metadata = []
        self.bm25 = None

    def load_chunks(self):
        """Load chunk texts and metadata"""
        for file_name in os.listdir(CHUNKS_DIR):
            if not file_name.endswith(".json"):
                continue

            with open(os.path.join(CHUNKS_DIR, file_name), "r", encoding="utf-8") as f:
                chunk = json.load(f)

            tokens = chunk["text"].lower().split()

            self.corpus.append(tokens)
            self.chunk_metadata.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "url": chunk["url"],
                "title": chunk["title"],
                "text": chunk["text"]
            })

        print(f"Loaded {len(self.corpus)} chunks for BM25")

    def build_index(self):
        """Build BM25 index"""
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query, top_k=10):
        """Retrieve top-k chunks for a query"""
        if self.bm25 is None:
            raise ValueError("BM25 index not built")

        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            meta = self.chunk_metadata[idx]
            results.append({
                "rank": rank,
                "score": float(scores[idx]),
                "chunk_id": meta["chunk_id"],
                "doc_id": meta["doc_id"],
                "url": meta["url"],
                "title": meta["title"],
                "text": meta["text"]
            })

        return results


if __name__ == "__main__":
    retriever = BM25Retriever()
    retriever.load_chunks()
    retriever.build_index()

    query = "How was bengal important for India in terms of Computer Science?"
    results = retriever.retrieve(query, top_k=5)

    for r in results:
        print(f"[Rank {r['rank']}] {r['url']} | Score: {r['score']:.4f}")
