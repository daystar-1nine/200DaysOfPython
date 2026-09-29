"""
Evaluation metrics, perplexity, and repetition analysis for MiniGPT.
"""
from app.evaluation.metrics import compute_perplexity, compute_distinct_n, evaluate_text_diversity
from app.evaluation.repetition import repetition_rate, analyze_repetition

__all__ = [
    "compute_perplexity",
    "compute_distinct_n",
    "evaluate_text_diversity",
    "repetition_rate",
    "analyze_repetition"
]
