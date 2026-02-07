import time
import json

from src.retrieval.dense import DenseRetriever
from src.retrieval.BM25_sparse import BM25Retriever
from src.retrieval.rrf import reciprocal_rank_fusion

from src.generation.generator import ResponseGenerator

from src.evaluation.mrr import mean_reciprocal_rank
from src.evaluation.recall import recall_at_k
from src.evaluation.semantic_similarity import SemanticSimilarity


QUESTIONS_FILE = "data/questions/questions_100.json"


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_mode(mode, questions, dense, sparse, generator, sim_metric):

    retrieved_all = []
    gold_urls_all = []

    pred_answers = []
    gold_answers = []

    start = time.time()

    for q in questions:

        dense_results = dense.retrieve(q["question"], top_k=10)
        sparse_results = sparse.retrieve(q["question"], top_k=10)

        if mode == "dense":
            retrieved = dense_results

        elif mode == "sparse":
            retrieved = sparse_results

        else:
            retrieved = reciprocal_rank_fusion(
                [dense_results, sparse_results],
                k=60,
                top_n=5
            )

        response = generator.generate(q["question"], retrieved)

        retrieved_all.append(retrieved)
        gold_urls_all.append(q["gold_urls"])

        pred_answers.append(response["answer"])
        gold_answers.append(q["ground_truth_answer"])

    elapsed = time.time() - start

    results = {
        "MRR": mean_reciprocal_rank(retrieved_all, gold_urls_all),
        "Recall@5": recall_at_k(retrieved_all, gold_urls_all, 5),
        "SemanticSimilarity": sim_metric.average_score(
            pred_answers,
            gold_answers
        ),
        "AvgResponseTime": elapsed / len(questions)
    }

    return results


def main():

    print("\n=== Ablation Study ===\n")

    questions = load_questions()

    dense = DenseRetriever()
    dense.build_index()

    sparse = BM25Retriever()
    sparse.load_chunks()
    sparse.build_index()

    generator = ResponseGenerator()
    sim_metric = SemanticSimilarity()

    modes = ["dense", "sparse", "hybrid"]

    summary = {}

    for mode in modes:
        print(f"Running {mode.upper()}...")
        summary[mode] = evaluate_mode(
            mode,
            questions,
            dense,
            sparse,
            generator,
            sim_metric
        )

    print("\n=== RESULTS ===")

    for mode, metrics in summary.items():

        print(f"\n{mode.upper()}")

        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
