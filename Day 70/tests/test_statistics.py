"""
Unit tests for summary statistics calculations.
"""

import math
import numpy as np
import pytest

try:
    from app.stats_calc import compute_sample_stats
except (ImportError, ModuleNotFoundError):
    from stats_calc import compute_sample_stats

def test_compute_sample_stats_basic():
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    res = compute_sample_stats(data)
    assert res["n"] == 5
    assert res["mean"] == 30.0
    assert res["df"] == 4
    assert res["min"] == 10.0
    assert res["max"] == 50.0
    assert res["median"] == 30.0
    # Sample variance: sum((x - 30)^2)/4 = (400 + 100 + 0 + 100 + 400)/4 = 1000/4 = 250
    assert math.isclose(res["variance"], 250.0)
    assert math.isclose(res["std"], math.sqrt(250.0))
    assert math.isclose(res["se"], math.sqrt(250.0) / math.sqrt(5))

def test_compute_sample_stats_degrees_of_freedom():
    data = list(range(20))
    res = compute_sample_stats(data)
    assert res["df"] == 19
