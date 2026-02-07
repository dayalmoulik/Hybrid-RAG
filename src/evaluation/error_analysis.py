import json
import pandas as pd

from src.retrieval.dense import DenseRetriever
from src.retrieval.BM25_sparse import BM25Retriever
from src.retrieval.rrf import reciprocal_rank_fusion
from src.generation.generator import ResponseGenerator
from src.evaluation.semantic_similarity import SemanticSimilarity

QUESTIONS_FILE = "data/questions/questions_100.json"
OUTPUT_CSV = "data/error_analysis.csv"


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def url_ranking(chunks):
    seen = set()
    ranking = []

    for c in chunks:
        if c["url"] not in seen:
            ranking.append(c["url"])
            seen.add(c["url"])

    return ranking


def classify_error(url_rank, gold_urls, sim_score):

    if not any(u in url_rank for u in gold_urls):
        return "Retrieval miss"

    first_hit = next(i for i, u in enumerate(url_rank) if u in gold_urls)

    if first_hit > 3:
        return "Ranking failure"

    if sim_score < 0.6:
        return "Generation failure"

    return "Success"


def main():

    print("\nRunning error analysis...\n")

    questions = load_questions()

    dense = DenseRetriever()
    dense.build_index()

    sparse = BM25Retriever()
    sparse.load_chunks()
    sparse.build_index()

    generator = ResponseGenerator()
    sim = SemanticSimilarity()

    records = []

    for q in questions:

        dense_results = dense.retrieve(q["question"], top_k=10)
        sparse_results = sparse.retrieve(q["question"], top_k=10)

        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=60,
            top_n=5
        )

        response = generator.generate(q["question"], fused)

        ranking = url_ranking(fused)

        sim_score = sim.score_pair(
            response["answer"],
            q["ground_truth_answer"]
        )

        error_type = classify_error(
            ranking,
            q["gold_urls"],
            sim_score
        )

        records.append({
            "Question": q["question"],
            "Category": q["category"],
            "PredictedAnswer": response["answer"],
            "GroundTruth": q["ground_truth_answer"],
            "Similarity": sim_score,
            "ErrorType": error_type
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)

    print("Saved error report →", OUTPUT_CSV)

    print("\nError summary:")
    print(df["ErrorType"].value_counts())


if __name__ == "__main__":
    main()
