"""
Tests for RandomSampler.
"""
import pytest
import numpy as np
from app.sampler import RandomSampler

def test_sample_shape(normal_population, sampler):
    sample = sampler.sample(normal_population, size=50)
    assert len(sample) == 50

def test_sample_without_replacement_uniqueness():
    sampler = RandomSampler(seed=42)
    pop = np.arange(100)
    sample = sampler.sample(pop, size=100, replace=False)
    assert len(sample) == 100
    assert len(np.unique(sample)) == 100

def test_sample_with_replacement_allows_duplicates():
    sampler = RandomSampler(seed=42)
    pop = np.array([1, 2, 3])
    sample = sampler.sample(pop, size=100, replace=True)
    assert len(sample) == 100
    assert len(np.unique(sample)) <= 3

def test_sample_exceeds_population_without_replacement(sampler):
    pop = np.arange(10)
    with pytest.raises(ValueError, match="exceeds population size"):
        sampler.sample(pop, size=15, replace=False)

def test_sample_invalid_size(sampler, normal_population):
    with pytest.raises(ValueError, match="strictly positive"):
        sampler.sample(normal_population, size=0)

def test_sample_multiple(sampler, normal_population):
    samples = sampler.sample_multiple(normal_population, size=20, n_samples=5)
    assert len(samples) == 5
    for s in samples:
        assert len(s) == 20

def test_sample_multiple_invalid_n_samples(sampler, normal_population):
    with pytest.raises(ValueError, match="n_samples must be strictly positive"):
        sampler.sample_multiple(normal_population, size=20, n_samples=-1)
