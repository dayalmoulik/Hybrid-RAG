import json
from src.retrieval.dense import DenseRetriever
from src.retrieval.BM25_sparse import BM25Retriever
from src.retrieval.rrf import reciprocal_rank_fusion
from src.evaluation.mrr import mean_reciprocal_rank
from src.evaluation.recall import recall_at_k
from src.generation.generator import ResponseGenerator
from src.evaluation.semantic_similarity import SemanticSimilarity



QUESTIONS_FILE = "data/questions/questions_100.json"


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_mrr(mode="hybrid"):
    questions = load_questions()

    dense = DenseRetriever()
    dense.build_index()

    sparse = BM25Retriever()
    sparse.load_chunks()
    sparse.build_index()

    all_retrieved = []
    all_gold_urls = []

    for q in questions:
        query = q["question"]

        dense_results = dense.retrieve(query, top_k=10)
        sparse_results = sparse.retrieve(query, top_k=10)

        if mode == "dense":
            retrieved = dense_results
        elif mode == "sparse":
            retrieved = sparse_results
        else:
            retrieved = reciprocal_rank_fusion(
                [dense_results, sparse_results],
                k=60,
                top_n=10
            )

        all_retrieved.append(retrieved)
        all_gold_urls.append(q["gold_urls"])

    return mean_reciprocal_rank(all_retrieved, all_gold_urls)


if __name__ == "__main__":

    questions = load_questions()

    dense = DenseRetriever()
    dense.build_index()

    sparse = BM25Retriever()
    sparse.load_chunks()
    sparse.build_index()

    for mode in ["dense", "sparse", "hybrid"]:
        generator = ResponseGenerator()
        sim_metric = SemanticSimilarity()

        pred_answers = []
        gold_answers = []
        all_retrieved = []
        all_gold = []

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
    
            all_retrieved.append(retrieved)
            all_gold.append(q["gold_urls"])
        
            response = generator.generate(q["question"], retrieved)

            pred_answers.append(response["answer"])
            gold_answers.append(q["ground_truth_answer"])

        mrr = mean_reciprocal_rank(all_retrieved, all_gold)
        recall = recall_at_k(all_retrieved, all_gold, k=5)
        similarity = sim_metric.average_score(pred_answers, gold_answers)

        print(f"\n{mode.upper()} RESULTS")
        print(f"MRR: {mrr:.4f}")
        print(f"Recall@5: {recall:.4f}")
        print(f"Semantic similarity: {similarity:.4f}")
        