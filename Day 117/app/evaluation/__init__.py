"""
Evaluation and benchmarking package for Day 117 Mini RAG system.
"""
from app.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    hit_rate_at_k,
    f1_at_k
)
from app.evaluation.grounding import (
    compute_faithfulness,
    compute_context_relevance,
    evaluate_grounding
)
from app.evaluation.evaluator import RAGEvaluator

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "hit_rate_at_k",
    "f1_at_k",
    "compute_faithfulness",
    "compute_context_relevance",
    "evaluate_grounding",
    "RAGEvaluator"
]
