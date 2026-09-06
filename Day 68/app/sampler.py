"""
Random sampling mechanism.
"""
import numpy as np

class RandomSampler:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def sample(self, population: np.ndarray, size: int, replace: bool = False) -> np.ndarray:
        if size <= 0:
            raise ValueError("Sample size must be strictly positive.")
        if not replace and size > len(population):
            raise ValueError(f"Sample size {size} exceeds population size {len(population)} without replacement.")
        return self.rng.choice(population, size=size, replace=replace)

    def sample_multiple(self, population: np.ndarray, size: int, n_samples: int, replace: bool = False) -> list[np.ndarray]:
        if n_samples <= 0:
            raise ValueError("n_samples must be strictly positive.")
        return [self.sample(population, size=size, replace=replace) for _ in range(n_samples)]
