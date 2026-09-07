"""
Tests for sample_size module.
"""
import pytest
from app.sample_size import SampleSizeCalculator

def test_sample_size_mean_calculation():
    # sigma=20, error=2, conf=0.95 -> exact=384.16 -> ceil=385
    n = SampleSizeCalculator.for_mean(population_std=20.0, margin_of_error=2.0, confidence=0.95)
    assert n == 385

def test_sample_size_mean_invalid_std():
    with pytest.raises(ValueError, match="strictly positive"):
        SampleSizeCalculator.for_mean(population_std=-5.0, margin_of_error=2.0)

def test_sample_size_proportion_worst_case():
    # E=0.03, conf=0.95, p=0.5 -> (1.960^2 * 0.25) / 0.0009 ~ 1067.11 -> 1068
    n = SampleSizeCalculator.for_proportion(margin_of_error=0.03, confidence=0.95, estimated_p=0.5)
    assert n == 1068

def test_sample_size_scales_quadratically_with_error():
    n_e2 = SampleSizeCalculator.for_mean(population_std=20.0, margin_of_error=2.0, confidence=0.95)
    n_e1 = SampleSizeCalculator.for_mean(population_std=20.0, margin_of_error=1.0, confidence=0.95)
    # Halving error from 2 to 1 quadruples required sample size
    ratio = n_e1 / n_e2
    assert 3.9 <= ratio <= 4.1
