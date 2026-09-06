"""
Unit tests for Bayesian calculations.
"""
import pytest
from app.probability.bayes import bayes_theorem, bayes_update

def test_bayes_theorem_standard():
    # P(B|A) = 0.8, P(A) = 0.1, P(B) = 0.2 -> P(A|B) = 0.8 * 0.1 / 0.2 = 0.4
    assert bayes_theorem(0.8, 0.1, 0.2) == pytest.approx(0.4)

def test_bayes_theorem_invalid_inputs():
    with pytest.raises(ValueError):
        bayes_theorem(0.8, 0.1, 0.0)

def test_bayes_update_medical():
    # Disease prior = 0.01, Sensitivity = 0.95, False Positive = 0.05
    res = bayes_update(prior=0.01, likelihood_true=0.95, likelihood_false=0.05)
    # Evidence = 0.95*0.01 + 0.05*0.99 = 0.0095 + 0.0495 = 0.0590
    assert res["marginal_evidence"] == pytest.approx(0.059, abs=1e-4)
    # Posterior = 0.0095 / 0.0590 = 0.161017
    assert res["posterior"] == pytest.approx(0.161017, abs=1e-4)

def test_bayes_update_edge_cases():
    # Certain prior
    res1 = bayes_update(prior=1.0, likelihood_true=0.9, likelihood_false=0.1)
    assert res1["posterior"] == pytest.approx(1.0)
    
    # Zero prior
    res0 = bayes_update(prior=0.0, likelihood_true=0.9, likelihood_false=0.1)
    assert res0["posterior"] == pytest.approx(0.0)
