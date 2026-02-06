from typing import List, Dict


def compute_url_ranking(retrieved_chunks: List[Dict]) -> List[str]:
    """
    Convert chunk ranking to URL ranking.
    Keeps first occurrence of each URL.
    """
    seen = set()
    ranking = []

    for item in retrieved_chunks:
        url = item["url"]
        if url not in seen:
            ranking.append(url)
            seen.add(url)

    return ranking


def recall_at_k(
    all_retrieved_chunks: List[List[Dict]],
    all_gold_urls: List[List[str]],
    k: int = 5
    ) -> float:
    """
    Compute URL-level Recall@K across queries.
    """

    assert len(all_retrieved_chunks) == len(all_gold_urls)

    hits = 0

    for retrieved, gold in zip(all_retrieved_chunks, all_gold_urls):

        url_ranking = compute_url_ranking(retrieved)[:k]

        if any(url in gold for url in url_ranking):
            hits += 1

    return hits / len(all_retrieved_chunks)
