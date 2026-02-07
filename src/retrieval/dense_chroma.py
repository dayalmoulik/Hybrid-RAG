import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


DB_DIR = "chroma_db"


class ChromaDenseRetriever:

    def __init__(self):

        print("Loading Chroma dense retriever...")

        embed_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        client = chromadb.PersistentClient(path=DB_DIR)

        self.collection = client.get_collection(
            name="rag_chunks",
            embedding_function=embed_fn
        )

    def retrieve(self, query, top_k=5):

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        output = []

        for i in range(len(results["documents"][0])):

            output.append({
                "rank": i + 1,
                "score": 1.0 - results["distances"][0][i],
                "text": results["documents"][0][i],
                "url": results["metadatas"][0][i]["url"],
                "title": results["metadatas"][0][i]["title"]
            })

        return output


if __name__ == "__main__":

    retriever = ChromaDenseRetriever()

    query = "How bengal was important for India in terms of Computer Science?"

    results = retriever.retrieve(query)

    for r in results:
        print(f"[Rank {r['rank']}] {r['url']} — score: {r['score']:.4f}")
