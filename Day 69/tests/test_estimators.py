"""
Tests for PointEstimator module.
"""
import pytest
import numpy as np
from app.estimators import PointEstimator

def test_estimate_mean(exact_mean_data):
    assert PointEstimator.estimate_mean(exact_mean_data) == 30.0

def test_estimate_variance_and_std(exact_mean_data):
    var = PointEstimator.estimate_variance(exact_mean_data, ddof=1)
    std = PointEstimator.estimate_std(exact_mean_data, ddof=1)
    assert var == 250.0
    assert np.isclose(std, np.sqrt(250.0))

def test_estimate_proportion():
    p = PointEstimator.estimate_proportion(320, 500)
    assert p == 0.64
