"""
Population synthesis with known theoretical parameters.
"""
import numpy as np

class PopulationGenerator:
    def __init__(self, size: int = 50_000, seed: int = 42):
        self.size = size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_normal(self, mean: float = 100.0, std: float = 15.0) -> np.ndarray:
        return self.rng.normal(loc=mean, scale=std, size=self.size)

    def generate_skewed(self, rate: float = 0.02) -> np.ndarray:
        # Exponential distribution with mean = 1/rate (e.g. 50.0)
        return self.rng.exponential(scale=1.0 / rate, size=self.size)

    def generate_uniform(self, low: float = 0.0, high: float = 100.0) -> np.ndarray:
        return self.rng.uniform(low=low, high=high, size=self.size)

    def generate_bimodal(self, mean1: float = 30.0, std1: float = 5.0,
                         mean2: float = 70.0, std2: float = 8.0,
                         weight1: float = 0.5) -> np.ndarray:
        n1 = int(self.size * weight1)
        n2 = self.size - n1
        part1 = self.rng.normal(loc=mean1, scale=std1, size=n1)
        part2 = self.rng.normal(loc=mean2, scale=std2, size=n2)
        pop = np.concatenate([part1, part2])
        self.rng.shuffle(pop)
        return pop
