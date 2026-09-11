"""
Unit tests for Cohen's d effect size computation and tiers.
"""

import math
import pytest

try:
    from app.effect_size import compute_cohens_d
except (ImportError, ModuleNotFoundError):
    from effect_size import compute_cohens_d

def test_cohens_d_negligible():
    res = compute_cohens_d(sample_mean=101.0, mu_0=100.0, sample_std=10.0)
    assert math.isclose(res["cohens_d"], 0.1)
    assert res["magnitude"] == "Negligible"

def test_cohens_d_small():
    res = compute_cohens_d(sample_mean=103.5, mu_0=100.0, sample_std=10.0)
    assert math.isclose(res["cohens_d"], 0.35)
    assert res["magnitude"] == "Small"

def test_cohens_d_medium():
    res = compute_cohens_d(sample_mean=106.0, mu_0=100.0, sample_std=10.0)
    assert math.isclose(res["cohens_d"], 0.6)
    assert res["magnitude"] == "Medium"

def test_cohens_d_large():
    res = compute_cohens_d(sample_mean=112.0, mu_0=100.0, sample_std=10.0)
    assert math.isclose(res["cohens_d"], 1.2)
    assert res["magnitude"] == "Large"

def test_cohens_d_negative_difference():
    res = compute_cohens_d(sample_mean=90.0, mu_0=100.0, sample_std=10.0)
    assert math.isclose(res["cohens_d"], -1.0)
    assert res["abs_cohens_d"] == 1.0
    assert res["magnitude"] == "Large"

def test_cohens_d_invalid_std():
    with pytest.raises(ValueError):
        compute_cohens_d(sample_mean=100.0, mu_0=100.0, sample_std=0.0)
