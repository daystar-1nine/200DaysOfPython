"""
Configuration settings and paths for Day 69 Capstone Analyzer.
"""
from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"

# Ensure directories exist
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class AppConfig:
    random_seed: int = 42
    default_confidence: float = 0.95
    confidence_levels: tuple[float, ...] = (0.90, 0.95, 0.99)
    sample_sizes: tuple[int, ...] = (10, 30, 50, 100, 250, 500, 1000)
    bootstrap_iterations: int = 10_000
    data_file: Path = DATA_DIR / "customer_orders.csv"
    output_dir: Path = OUTPUT_DIR
    charts_dir: Path = CHARTS_DIR
    ci_results_file: Path = OUTPUT_DIR / "confidence_intervals.csv"
    bootstrap_results_file: Path = OUTPUT_DIR / "bootstrap_results.csv"
    report_file: Path = OUTPUT_DIR / "estimation_report.txt"
