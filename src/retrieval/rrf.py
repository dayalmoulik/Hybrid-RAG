from collections import defaultdict


def reciprocal_rank_fusion(result_lists, k=60, top_n=5):
    """
    Perform Reciprocal Rank Fusion (RRF)

    Args:
        result_lists (list): List of retrieval result lists.
                             Each result list is a list of dicts with 'rank'.
        k (int): RRF constant (default: 60)
        top_n (int): Number of fused results to return

    Returns:
        List of fused results sorted by RRF score
    """
    scores = defaultdict(float)
    metadata = {}

    for results in result_lists:
        for item in results:
            rank = item["rank"]
            chunk_id = item["chunk_id"]

            scores[chunk_id] += 1.0 / (k + rank)

            # Store metadata once
            if chunk_id not in metadata:
                metadata[chunk_id] = item

    fused = []
    for chunk_id, score in scores.items():
        item = metadata[chunk_id].copy()
        item["rrf_score"] = score
        fused.append(item)

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    # Assign fused ranks
    for idx, item in enumerate(fused[:top_n], start=1):
        item["rank"] = idx

    return fused[:top_n]


if __name__ == "__main__":
    # Minimal test with fake ranks
    dense_results = [
        {"chunk_id": "c1", "rank": 1, "url": "url1"},
        {"chunk_id": "c2", "rank": 2, "url": "url2"},
    ]

    sparse_results = [
        {"chunk_id": "c2", "rank": 1, "url": "url2"},
        {"chunk_id": "c3", "rank": 2, "url": "url3"},
    ]

    fused = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        k=60,
        top_n=3
    )

    for f in fused:
        print(f)
