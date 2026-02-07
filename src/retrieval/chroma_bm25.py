import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi


DB_DIR = "chroma_db"


class ChromaBM25Retriever:

    def __init__(self):

        embed_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        client = chromadb.PersistentClient(path=DB_DIR)

        self.collection = client.get_collection(
            name="rag_chunks",
            embedding_function=embed_fn
        )

        self._load_chunks()

    def _load_chunks(self):

        print("Loading Chroma chunks for BM25...")

        data = self.collection.get()

        self.texts = data["documents"]
        self.meta = data["metadatas"]

        tokenized = [t.lower().split() for t in self.texts]

        self.bm25 = BM25Okapi(tokenized)

        print(f"Loaded {len(self.texts)} chunks")

    def retrieve(self, query, top_k=5):

        tokens = query.lower().split()

        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for rank, idx in enumerate(ranked, start=1):

            results.append({
                "rank": rank,
                "score": float(scores[idx]),
                "text": self.texts[idx],
                "url": self.meta[idx]["url"],
                "title": self.meta[idx]["title"]
            })

        return results


if __name__ == "__main__":

    retriever = ChromaBM25Retriever()

    query = "How bengal was important for India in terms of Computer Science?"

    results = retriever.retrieve(query)

    for r in results:
        print(f"[Rank {r['rank']}] {r['url']}")