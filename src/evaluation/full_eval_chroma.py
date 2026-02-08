import json
import time

from src.retrieval.dense_chroma import ChromaDenseRetriever
from src.retrieval.chroma_bm25 import ChromaBM25Retriever
from src.retrieval.rrf import reciprocal_rank_fusion

from src.generation.generator import ResponseGenerator

from src.evaluation.mrr import mean_reciprocal_rank
from src.evaluation.recall import recall_at_k
from src.evaluation.semantic_similarity import SemanticSimilarity


QUESTIONS_FILE = "data/questions/questions_100.json"
OUTPUT_FILE = "data/chroma_eval_results.json"


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_mode(mode, questions, dense, bm25, generator, sim):

    retrieved_all = []
    gold_urls_all = []

    preds = []
    refs = []

    start = time.time()

    for q in questions:

        query = q["question"]

        dense_results = dense.retrieve(query, top_k=10)
        bm25_results = bm25.retrieve(query, top_k=10)

        if mode == "dense":
            retrieved = dense_results

        elif mode == "bm25":
            retrieved = bm25_results

        else:
            retrieved = reciprocal_rank_fusion(
                [dense_results, bm25_results],
                k=60,
                top_n=5
            )

        response = generator.generate(query, retrieved)

        retrieved_all.append(retrieved)
        gold_urls_all.append(q["gold_urls"])

        preds.append(response["answer"])
        refs.append(q["ground_truth_answer"])

    elapsed = time.time() - start

    return {
        "MRR": mean_reciprocal_rank(retrieved_all, gold_urls_all),
        "Recall@5": recall_at_k(retrieved_all, gold_urls_all, 5),
        "SemanticSimilarity": sim.average_score(preds, refs),
        "AvgResponseTime": elapsed / len(questions)
    }


def main():

    print("\n=== Full Chroma Evaluation Pipeline ===\n")

    questions = load_questions()

    dense = ChromaDenseRetriever()
    bm25 = ChromaBM25Retriever()

    generator = ResponseGenerator()
    sim = SemanticSimilarity()

    results = {}

    for mode in ["dense", "bm25", "hybrid"]:

        print(f"Evaluating {mode.upper()}...")

        results[mode] = evaluate_mode(
            mode,
            questions,
            dense,
            bm25,
            generator,
            sim
        )

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved results →", OUTPUT_FILE)

    print("\n=== SUMMARY ===")

    for mode, metrics in results.items():

        print(f"\n{mode.upper()}")

        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()