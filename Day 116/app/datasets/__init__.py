"""
Datasets module for Day 116 evaluation harness.
"""
from app.datasets.schema import EvaluationExample, VALID_CATEGORIES, VALID_TYPES
from app.datasets.loader import (
    load_jsonl_dataset,
    save_jsonl_dataset,
    filter_by_category,
    validate_dataset,
    batch_examples
)
from app.datasets.splits import stratified_split, detect_data_leakage

__all__ = [
    "EvaluationExample",
    "VALID_CATEGORIES",
    "VALID_TYPES",
    "load_jsonl_dataset",
    "load_evaluation_dataset",
    "save_jsonl_dataset",
    "filter_by_category",
    "validate_dataset",
    "batch_examples",
    "stratified_split",
    "detect_data_leakage"
]
