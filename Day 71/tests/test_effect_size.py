"""
Unit tests for Cohen's d effect sizes.
"""

import math
import pytest

try:
    from app.effect_size import compute_cohens_d_independent, compute_cohens_d_paired
except (ImportError, ModuleNotFoundError):
    from effect_size import compute_cohens_d_independent, compute_cohens_d_paired

def test_cohens_d_independent():
    a = [10.0, 11.0, 10.5, 11.5, 10.0]
    b = [15.0, 16.0, 15.5, 16.5, 15.0]
    d_res = compute_cohens_d_independent(a, b)
    assert d_res["cohens_d"] > 2.0
    assert d_res["magnitude"] == "Large"

def test_cohens_d_paired():
    before = [50.0, 52.0, 51.0, 49.0, 50.5]
    after = [50.5, 52.3, 51.4, 49.2, 50.8]  # Very small difference
    d_res = compute_cohens_d_paired(before, after)
    assert d_res["abs_cohens_d"] > 0
