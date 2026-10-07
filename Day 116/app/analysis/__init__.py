"""
Statistical analysis, model comparison, and error taxonomy classification.
"""
from app.analysis.statistics import (
    bootstrap_confidence_interval,
    paired_permutation_test,
    paired_t_test,
    compute_cohens_d
)
from app.analysis.error_analysis import (
    classify_failure_mode,
    run_error_analysis,
    FAILURE_TYPES
)
from app.analysis.comparison import (
    ModelComparator,
    compare_benchmark_results
)

__all__ = [
    "bootstrap_confidence_interval",
    "paired_permutation_test",
    "paired_t_test",
    "compute_cohens_d",
    "classify_failure_mode",
    "run_error_analysis",
    "FAILURE_TYPES",
    "ModelComparator",
    "compare_benchmark_results"
]
