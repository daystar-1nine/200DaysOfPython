"""
Configuration dataclasses and paths for Day 70 Hypothesis Testing Engine.
"""

from dataclasses import dataclass
from enum import Enum
import os

class AlternativeType(str, Enum):
    TWO_SIDED = "two-sided"
    GREATER = "greater"
    LESS = "less"

class TestKind(str, Enum):
    __test__ = False
    ONE_SAMPLE_T = "one_sample_t"
    ONE_SAMPLE_Z = "one_sample_z"
    PROPORTION_Z = "proportion_z"

@dataclass
class AppConfig:
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")
    CHARTS_DIR: str = os.path.join(OUTPUT_DIR, "charts")
    RESULTS_CSV: str = os.path.join(OUTPUT_DIR, "hypothesis_test_results.csv")
    REPORT_TXT: str = os.path.join(OUTPUT_DIR, "statistical_report.txt")
    
    DEFAULT_ALPHA: float = 0.05
    RANDOM_SEED: int = 42

config = AppConfig()
