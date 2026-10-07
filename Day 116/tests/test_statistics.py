"""
Unit tests for bootstrap confidence intervals, paired permutation tests, and effect sizes.
"""
import pytest
import numpy as np

from app.analysis.statistics import (
    bootstrap_confidence_interval,
    paired_permutation_test,
    paired_t_test,
    compute_cohens_d
)


def test_bootstrap_confidence_interval_mean_accuracy():
    scores = [10.0, 20.0, 30.0, 40.0, 50.0]
    res = bootstrap_confidence_interval(scores, n_bootstrap=1000, seed=42)
    assert res["mean"] == 30.0
    assert res["ci_lower"] < 30.0
    assert res["ci_upper"] > 30.0


def test_bootstrap_confidence_interval_ci_contains_mean():
    rng = np.random.RandomState(42)
    scores = rng.normal(loc=75.0, scale=5.0, size=100).tolist()
    res = bootstrap_confidence_interval(scores, n_bootstrap=1000, ci=0.95, seed=42)
    assert res["ci_lower"] <= res["mean"] <= res["ci_upper"]
    assert res["std_err"] > 0.0


def test_bootstrap_confidence_interval_single_value():
    res = bootstrap_confidence_interval([42.5])
    assert res["mean"] == 42.5
    assert res["ci_lower"] == 42.5
    assert res["ci_upper"] == 42.5
    assert res["std_err"] == 0.0


def test_bootstrap_confidence_interval_empty_list():
    res = bootstrap_confidence_interval([])
    assert res["mean"] == 0.0
    assert res["ci_lower"] == 0.0
    assert res["ci_upper"] == 0.0


def test_bootstrap_confidence_interval_zero_variance():
    scores = [50.0] * 20
    res = bootstrap_confidence_interval(scores, n_bootstrap=500)
    assert res["mean"] == 50.0
    assert res["ci_lower"] == 50.0
    assert res["ci_upper"] == 50.0


def test_paired_permutation_test_identical_distributions():
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    res = paired_permutation_test(scores, scores, n_permutations=500)
    assert res["observed_diff"] == 0.0
    assert res["p_value"] == 1.0
    assert res["is_significant"] is False


def test_paired_permutation_test_significant_difference():
    scores_a = [90.0, 92.0, 88.0, 95.0, 91.0, 93.0, 89.0, 94.0]
    scores_b = [20.0, 22.0, 25.0, 21.0, 23.0, 19.0, 24.0, 22.0]
    res = paired_permutation_test(scores_a, scores_b, n_permutations=1000)
    assert res["observed_diff"] > 60.0
    assert res["p_value"] < 0.05
    assert res["is_significant"] is True


def test_paired_permutation_test_length_mismatch_raises():
    with pytest.raises(ValueError, match="Score arrays must have equal length"):
        paired_permutation_test([1.0, 2.0], [1.0])


def test_paired_permutation_test_empty_lists():
    res = paired_permutation_test([], [])
    assert res["observed_diff"] == 0.0
    assert res["is_significant"] is False


def test_paired_t_test_identical_distributions():
    scores = [10.0, 20.0, 30.0]
    res = paired_t_test(scores, scores)
    assert res["t_stat"] == 0.0
    assert res["is_significant"] is False


def test_paired_t_test_significant_difference():
    a = [85.0, 90.0, 88.0, 92.0, 89.0, 91.0, 87.0]
    b = [40.0, 42.0, 41.0, 39.0, 43.0, 40.0, 41.0]
    res = paired_t_test(a, b)
    assert res["t_stat"] > 10.0
    assert res["p_value"] < 0.001
    assert res["is_significant"] is True


def test_paired_t_test_length_mismatch_raises():
    with pytest.raises(ValueError, match="Score lengths must match"):
        paired_t_test([1.0], [1.0, 2.0])


def test_compute_cohens_d_zero_difference():
    scores = [10.0, 20.0, 30.0]
    d = compute_cohens_d(scores, scores)
    assert d == 0.0


def test_compute_cohens_d_large_effect():
    a = [100.0, 102.0, 101.0, 103.0, 99.0]
    b = [50.0, 51.0, 49.0, 52.0, 48.0]
    d = compute_cohens_d(a, b)
    assert d > 10.0  # Immense positive effect


def test_compute_cohens_d_small_sample_returns_zero():
    assert compute_cohens_d([1.0], [1.0]) == 0.0
    assert compute_cohens_d([], []) == 0.0
