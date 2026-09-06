"""
Tests for PopulationGenerator.
"""
import pytest
import numpy as np
from app.population import PopulationGenerator

def test_population_generator_initialization():
    gen = PopulationGenerator(size=5_000, seed=123)
    assert gen.size == 5_000
    assert gen.seed == 123

def test_generate_normal(normal_population):
    assert len(normal_population) == 10_000
    assert np.isclose(np.mean(normal_population), 100.0, atol=1.0)
    assert np.isclose(np.std(normal_population), 15.0, atol=1.0)

def test_generate_skewed(skewed_population):
    assert len(skewed_population) == 10_000
    assert np.all(skewed_population >= 0)
    # Exponential mean = 1/0.02 = 50.0
    assert np.isclose(np.mean(skewed_population), 50.0, atol=2.0)

def test_generate_uniform(uniform_population):
    assert len(uniform_population) == 10_000
    assert np.all(uniform_population >= 0.0)
    assert np.all(uniform_population <= 100.0)
    assert np.isclose(np.mean(uniform_population), 50.0, atol=1.5)

def test_generate_bimodal(bimodal_population):
    assert len(bimodal_population) == 10_000
    # Average of 30 and 70 with equal weights is 50
    assert np.isclose(np.mean(bimodal_population), 50.0, atol=1.5)

def test_reproducibility():
    gen1 = PopulationGenerator(size=1_000, seed=99)
    gen2 = PopulationGenerator(size=1_000, seed=99)
    p1 = gen1.generate_normal()
    p2 = gen2.generate_normal()
    np.testing.assert_array_equal(p1, p2)
