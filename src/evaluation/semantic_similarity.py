from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSimilarity:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def score_pair(self, pred: str, gold: str) -> float:
        embeddings = self.model.encode([pred, gold])
        sim = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]
        return float(sim)

    def average_score(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:

        scores = [
            self.score_pair(p, r)
            for p, r in zip(predictions, references)
        ]

        return float(np.mean(scores))
