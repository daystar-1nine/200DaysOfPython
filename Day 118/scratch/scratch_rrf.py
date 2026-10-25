"""
Challenge 1: Reciprocal Rank Fusion (RRF) from scratch.
Demonstrates multi-list rank aggregation without needing raw score calibration.
"""
from typing import List, Dict, Tuple
from collections import defaultdict


def rrf_fuse(
    ranked_lists: List[List[str]],
    k0: int = 60
) -> List[Tuple[str, float]]:
    """
    RRF Score: sum_{list in ranked_lists} 1 / (k0 + rank)
    where rank is 1-based position in each list.
    """
    scores = defaultdict(float)
    first_seen_rank = {}

    for rank_list in ranked_lists:
        for rank_1based, doc_id in enumerate(rank_list, start=1):
            scores[doc_id] += 1.0 / (k0 + rank_1based)
            if doc_id not in first_seen_rank or rank_1based < first_seen_rank[doc_id]:
                first_seen_rank[doc_id] = rank_1based

    # Sort descending by score; tie-break by earliest rank seen
    fused = sorted(scores.items(), key=lambda x: (x[1], -first_seen_rank[x[0]]), reverse=True)
    return fused


def main():
    print("=== Challenge 1: Reciprocal Rank Fusion (RRF) ===")
    
    # 3 heterogeneous rank lists (e.g. BM25, Dense Vector, Title Match)
    bm25_list = ["doc_gradient_descent", "doc_learning_rate", "doc_adam_optimizer", "doc_backprop"]
    dense_list = ["doc_adam_optimizer", "doc_gradient_descent", "doc_loss_function", "doc_learning_rate"]
    title_list = ["doc_learning_rate", "doc_gradient_descent", "doc_regularization"]

    fused = rrf_fuse([bm25_list, dense_list, title_list], k0=60)
    
    print("\nIndividual Ranked Lists:")
    print("  BM25: ", bm25_list)
    print("  Dense:", dense_list)
    print("  Title:", title_list)
    
    print("\nRRF Consolidated Ranking (k0=60):")
    for rank, (doc_id, score) in enumerate(fused, start=1):
        print(f"  Rank {rank}: {doc_id:<25} (RRF Score: {score:.5f})")

    assert fused[0][0] in {"doc_gradient_descent", "doc_learning_rate"}
    print("\n[SUCCESS] RRF ranking computed with stable tie-breaking!")


if __name__ == "__main__":
    main()
