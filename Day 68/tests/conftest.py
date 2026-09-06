"""
Pytest fixtures for Day 68 Inferential Statistics testing suite.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from app.population import PopulationGenerator
from app.sampler import RandomSampler


@pytest.fixture
def pop_generator():
    return PopulationGenerator(size=10_000, seed=42)

@pytest.fixture
def normal_population(pop_generator):
    return pop_generator.generate_normal(mean=100.0, std=15.0)

@pytest.fixture
def skewed_population(pop_generator):
    return pop_generator.generate_skewed(rate=0.02) # mean = 50.0

@pytest.fixture
def uniform_population(pop_generator):
    return pop_generator.generate_uniform(low=0.0, high=100.0)

@pytest.fixture
def bimodal_population(pop_generator):
    return pop_generator.generate_bimodal(mean1=30.0, std1=5.0, mean2=70.0, std2=8.0)

@pytest.fixture
def sampler():
    return RandomSampler(seed=42)
