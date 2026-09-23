"""
Evaluation metrics and error analysis module for Day 105.
"""

from .metrics import compute_classification_metrics
from .confusion_matrix import compute_confusion_matrix, format_confusion_matrix
from .error_analysis import ErrorAnalyzer

__all__ = [
    "compute_classification_metrics",
    "compute_confusion_matrix",
    "format_confusion_matrix",
    "ErrorAnalyzer"
]
