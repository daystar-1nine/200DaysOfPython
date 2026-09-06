"""
Comprehensive statistical analyzer orchestrating population, sampling, CLT, and bootstrap.
"""
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from app.config import SimulationConfig
    from app.population import PopulationGenerator
    from app.sampler import RandomSampler
    from app.sampling_distribution import SamplingDistributionSimulator
    from app.clt import CLTAnalyzer
    from app.bootstrap import BootstrapEstimator
except (ImportError, ModuleNotFoundError):
    from config import SimulationConfig
    from population import PopulationGenerator
    from sampler import RandomSampler
    from sampling_distribution import SamplingDistributionSimulator
    from clt import CLTAnalyzer
    from bootstrap import BootstrapEstimator


class SamplingAnalyzer:
    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self.pop_gen = PopulationGenerator(size=self.config.population_size, seed=self.config.random_seed)
        
        # Build synthetic populations
        self.populations = {
            "Normal": self.pop_gen.generate_normal(mean=100.0, std=15.0),
            "Exponential (Skewed)": self.pop_gen.generate_skewed(rate=0.02), # mean=50
            "Uniform": self.pop_gen.generate_uniform(low=0.0, high=100.0),    # mean=50
            "Bimodal": self.pop_gen.generate_bimodal(mean1=30.0, std1=5.0, mean2=70.0, std2=8.0)
        }

    def run_clt_analysis(self, pop_name: str = "Exponential (Skewed)") -> pd.DataFrame:
        pop = self.populations[pop_name]
        clt_analyzer = CLTAnalyzer(pop, seed=self.config.random_seed)
        return clt_analyzer.evaluate_convergence(self.config.sample_sizes, n_samples=self.config.n_simulations)

    def run_bootstrap_analysis(self, pop_name: str = "Exponential (Skewed)", sample_size: int = 50) -> dict:
        pop = self.populations[pop_name]
        sampler = RandomSampler(seed=self.config.random_seed)
        sample = sampler.sample(pop, size=sample_size, replace=False)
        estimator = BootstrapEstimator(sample, seed=self.config.random_seed)
        res = estimator.estimate(statistic_func=np.mean, n_iterations=self.config.bootstrap_iterations, ci_level=self.config.ci_level)
        res["sample"] = sample
        res["true_population_mean"] = float(np.mean(pop))
        return res
