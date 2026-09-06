"""
Unit and convergence tests for stochastic simulator.
"""
import pytest
from distributions.normal import NormalDistribution
from distributions.binomial import BinomialDistribution
from app.simulator import run_distribution_simulation

def test_simulation_sample_count():
    dist = NormalDistribution(mu=50, sigma=5)
    res = run_distribution_simulation(dist, n_samples=2_000, seed=42)
    assert len(res["samples"]) == 2_000
    assert len(res["metrics_df"]) == 6

def test_simulation_reproducibility():
    dist = BinomialDistribution(n=10, p=0.5)
    res1 = run_distribution_simulation(dist, n_samples=1_000, seed=123)
    res2 = run_distribution_simulation(dist, n_samples=1_000, seed=123)
    assert (res1["samples"] == res2["samples"]).all()

def test_simulation_convergence():
    dist = NormalDistribution(mu=100, sigma=15)
    res = run_distribution_simulation(dist, n_samples=25_000, seed=42)
    assert res["mean_converged_2se"] is True
