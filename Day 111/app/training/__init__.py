"""
Training and benchmarking routines for BERT spam classification.
"""
from app.training.trainer import BertTrainer
from app.training.fine_tuning import run_experiment_a_frozen, run_experiment_b_finetuned, compare_experiments
from app.training.benchmarking import run_tfidf_baseline, build_benchmark_table

__all__ = [
    "BertTrainer",
    "run_experiment_a_frozen",
    "run_experiment_b_finetuned",
    "compare_experiments",
    "run_tfidf_baseline",
    "build_benchmark_table"
]
