from src.retrieval.dense_chroma import ChromaDenseRetriever
from src.retrieval.chroma_bm25 import ChromaBM25Retriever
from src.retrieval.rrf import reciprocal_rank_fusion


def main():

    query = "How bengal was important for India in terms of Computer Science?"

    dense = ChromaDenseRetriever()
    bm25 = ChromaBM25Retriever()

    dense_results = dense.retrieve(query, top_k=10)
    sparse_results = bm25.retrieve(query, top_k=10)

    fused = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        k=60,
        top_n=5
    )

    print("\n=== Hybrid RRF Results ===\n")

    for r in fused:
        print(
            f"[Rank {r['rank']}] {r['url']} | "
            f"RRF score: {r['rrf_score']:.4f}"
        )


if __name__ == "__main__":
    main()