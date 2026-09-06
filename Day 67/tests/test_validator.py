"""
Unit tests for parameter validator.
"""
import pytest
from app.validator import validate_distribution_params

def test_validator_valid_inputs():
    validate_distribution_params("bernoulli", {"p": 0.5})
    validate_distribution_params("binomial", {"n": 10, "p": 0.3})
    validate_distribution_params("uniform", {"a": 0, "b": 1})
    validate_distribution_params("normal", {"mu": 0, "sigma": 1})
    validate_distribution_params("poisson", {"lambda": 4})

def test_validator_unknown_distribution():
    with pytest.raises(ValueError, match="Unsupported distribution"):
        validate_distribution_params("weibull", {})

def test_validator_missing_params():
    with pytest.raises(ValueError, match="requires parameter"):
        validate_distribution_params("bernoulli", {})
    with pytest.raises(ValueError, match="requires parameters"):
        validate_distribution_params("binomial", {"n": 10})
