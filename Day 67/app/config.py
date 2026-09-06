"""
Configuration settings and paths for Probability Distribution Analyzer.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Simulation Parameters
RANDOM_SEED = 42
DEFAULT_SAMPLE_SIZE = 50_000

# Canonical Distribution Profiles for Analysis
DEFAULT_DISTRIBUTIONS = {
    "bernoulli": {"p": 0.70},
    "binomial": {"n": 20, "p": 0.40},
    "uniform": {"a": 0.0, "b": 10.0},
    "normal": {"mu": 70.0, "sigma": 10.0},
    "poisson": {"lambda": 5.0}
}
