import json

from src.retrieval.dense_chroma import ChromaDenseRetriever
from src.retrieval.chroma_bm25 import ChromaBM25Retriever
from src.retrieval.rrf import reciprocal_rank_fusion
from src.evaluation.recall import recall_at_k


QUESTIONS_FILE = "data/questions/questions_100.json"


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(mode="chroma_dense", k=5):

    questions = load_questions()

    dense = ChromaDenseRetriever()
    bm25 = ChromaBM25Retriever()

    all_retrieved = []
    all_gold = []

    for q in questions:

        query = q["question"]

        dense_results = dense.retrieve(query, top_k=10)
        bm25_results = bm25.retrieve(query, top_k=10)

        if mode == "chroma_dense":
            retrieved = dense_results

        elif mode == "chroma_bm25":
            retrieved = bm25_results

        else:  # hybrid
            retrieved = reciprocal_rank_fusion(
                [dense_results, bm25_results],
                k=60,
                top_n=10
            )

        all_retrieved.append(retrieved)
        all_gold.append(q["gold_urls"])

    return recall_at_k(all_retrieved, all_gold, k)


if __name__ == "__main__":

    print("\n=== Chroma Recall@K Evaluation ===\n")

    for mode in ["chroma_dense", "chroma_bm25", "chroma_hybrid"]:

        score = evaluate(mode, k=5)

        print(f"{mode.upper()} Recall@5: {score:.4f}")