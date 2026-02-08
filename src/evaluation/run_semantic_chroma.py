import json

from src.retrieval.dense_chroma import ChromaDenseRetriever
from src.retrieval.chroma_bm25 import ChromaBM25Retriever
from src.retrieval.rrf import reciprocal_rank_fusion
from src.generation.generator import ResponseGenerator
from src.evaluation.semantic_similarity import SemanticSimilarity


QUESTIONS_FILE = "data/questions/questions_100.json"


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(mode="chroma_hybrid"):

    questions = load_questions()

    dense = ChromaDenseRetriever()
    bm25 = ChromaBM25Retriever()

    generator = ResponseGenerator()
    sim_metric = SemanticSimilarity()

    predictions = []
    references = []

    for q in questions:

        query = q["question"]

        dense_results = dense.retrieve(query, top_k=10)
        bm25_results = bm25.retrieve(query, top_k=10)

        if mode == "chroma_dense":
            retrieved = dense_results

        elif mode == "chroma_bm25":
            retrieved = bm25_results

        else:
            retrieved = reciprocal_rank_fusion(
                [dense_results, bm25_results],
                k=60,
                top_n=5
            )

        response = generator.generate(query, retrieved)

        predictions.append(response["answer"])
        references.append(q["ground_truth_answer"])

    return sim_metric.average_score(predictions, references)


if __name__ == "__main__":

    print("\n=== Chroma Semantic Similarity ===\n")

    for mode in ["chroma_dense", "chroma_bm25", "chroma_hybrid"]:

        score = evaluate(mode)

        print(f"{mode.upper()} similarity: {score:.4f}")
        