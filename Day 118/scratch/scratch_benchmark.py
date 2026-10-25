"""
Challenge 7: Information Retrieval Micro-Benchmark.
Calculates Precision@K, Recall@K, MRR, and nDCG@K on graded relevance judgments.
"""
import math
from typing import List, Dict, Set


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    top_k = retrieved[:k]
    return sum(1 for d in set(top_k) if d in relevant) / float(k)


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    top_k = retrieved[:k]
    return min(1.0, len(set(top_k).intersection(relevant)) / float(len(relevant)))


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    for rank, d in enumerate(retrieved, 1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], graded_rel: Dict[str, float], k: int) -> float:
    top_k = retrieved[:k]
    dcg = 0.0
    seen = set()
    for i, d in enumerate(top_k, 1):
        if d in seen:
            rel = 0.0
        else:
            rel = graded_rel.get(d, 0.0)
            seen.add(d)
        dcg += ((2.0 ** rel) - 1.0) / math.log2(i + 1.0)

    # IDCG
    ideal_grades = sorted(graded_rel.values(), reverse=True)[:k]
    idcg = sum(((2.0 ** rel) - 1.0) / math.log2(i + 1.0) for i, rel in enumerate(ideal_grades, 1))

    return min(1.0, dcg / idcg) if idcg > 0 else 0.0


def main():
    print("=== Challenge 7: IR Micro-Benchmark ===")
    
    # Ground truth for query "Python list comprehension memory"
    relevant_docs = {"doc_python_list", "doc_python_generators"}
    graded_relevance = {"doc_python_list": 3.0, "doc_python_generators": 2.0}

    # System 1: Baseline Retriever
    sys1_retrieved = ["doc_django", "doc_python_list", "doc_flask", "doc_python_generators", "doc_sql"]
    
    # System 2: Advanced Hybrid Retriever
    sys2_retrieved = ["doc_python_list", "doc_python_generators", "doc_iterators", "doc_django", "doc_flask"]

    print("\nSystem 1 (Baseline):", sys1_retrieved)
    print(f"  P@3:    {precision_at_k(sys1_retrieved, relevant_docs, 3):.4f}")
    print(f"  R@3:    {recall_at_k(sys1_retrieved, relevant_docs, 3):.4f}")
    print(f"  MRR:    {reciprocal_rank(sys1_retrieved, relevant_docs):.4f}")
    print(f"  nDCG@3: {ndcg_at_k(sys1_retrieved, graded_relevance, 3):.4f}")

    print("\nSystem 2 (Advanced Hybrid):", sys2_retrieved)
    print(f"  P@3:    {precision_at_k(sys2_retrieved, relevant_docs, 3):.4f}")
    print(f"  R@3:    {recall_at_k(sys2_retrieved, relevant_docs, 3):.4f}")
    print(f"  MRR:    {reciprocal_rank(sys2_retrieved, relevant_docs):.4f}")
    print(f"  nDCG@3: {ndcg_at_k(sys2_retrieved, graded_relevance, 3):.4f}")

    assert ndcg_at_k(sys2_retrieved, graded_relevance, 3) > ndcg_at_k(sys1_retrieved, graded_relevance, 3)
    print("\n[SUCCESS] IR Micro-benchmark metrics computed and validated!")


if __name__ == "__main__":
    main()
