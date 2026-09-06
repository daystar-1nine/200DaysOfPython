"""
Configuration settings and paths for Day 68 Capstone Simulator.
"""
from pathlib import Path
from dataclasses import dataclass, field

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"

# Ensure output directories exist
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class SimulationConfig:
    random_seed: int = 42
    population_size: int = 50_000
    sample_sizes: tuple[int, ...] = (5, 15, 30, 100)
    default_sample_size: int = 30
    n_simulations: int = 5_000
    bootstrap_iterations: int = 5_000
    ci_level: float = 0.95
    data_file: Path = DATA_DIR / "sample_data.csv"
    output_dir: Path = OUTPUT_DIR
    charts_dir: Path = CHARTS_DIR
    sampling_results_file: Path = OUTPUT_DIR / "sampling_results.csv"
    bootstrap_results_file: Path = OUTPUT_DIR / "bootstrap_results.csv"
    report_file: Path = OUTPUT_DIR / "statistics_report.txt"

