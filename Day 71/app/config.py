"""
Configuration dataclasses and paths for Day 71 A/B Testing Engine.
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
    INDEPENDENT_T = "independent_t"
    WELCH_T = "welch_t"
    PAIRED_T = "paired_t"
    TWO_PROPORTION_Z = "two_proportion_z"

@dataclass
class AppConfig:
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")
    CHARTS_DIR: str = os.path.join(OUTPUT_DIR, "charts")
    
    USERS_CSV: str = os.path.join(DATA_DIR, "experiment_users.csv")
    CONFIG_JSON: str = os.path.join(DATA_DIR, "experiment_config.json")
    
    EXP_SUMMARY_CSV: str = os.path.join(OUTPUT_DIR, "experiment_summary.csv")
    STATS_RESULTS_CSV: str = os.path.join(OUTPUT_DIR, "statistical_results.csv")
    BUSINESS_IMPACT_TXT: str = os.path.join(OUTPUT_DIR, "business_impact.txt")
    REPORT_TXT: str = os.path.join(OUTPUT_DIR, "ab_test_report.txt")
    
    DEFAULT_ALPHA: float = 0.05
    RANDOM_SEED: int = 42

config = AppConfig()
