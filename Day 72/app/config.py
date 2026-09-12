"""
Day 72 — Application Configuration and Enumerations
"""
import os
from enum import Enum
from dataclasses import dataclass

class StrengthTier(str, Enum):
    VERY_WEAK = "Very Weak (0.00 - 0.19)"
    WEAK = "Weak (0.20 - 0.39)"
    MODERATE = "Moderate (0.40 - 0.59)"
    STRONG = "Strong (0.60 - 0.79)"
    VERY_STRONG = "Very Strong (0.80 - 1.00)"

class DirectionType(str, Enum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    ZERO = "Zero"

def classify_strength(abs_r: float) -> StrengthTier:
    if abs_r < 0.20:
        return StrengthTier.VERY_WEAK
    elif abs_r < 0.40:
        return StrengthTier.WEAK
    elif abs_r < 0.60:
        return StrengthTier.MODERATE
    elif abs_r < 0.80:
        return StrengthTier.STRONG
    else:
        return StrengthTier.VERY_STRONG

@dataclass
class AppConfig:
    # Base directories
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH_CSV: str = os.path.join(BASE_DIR, "data", "ecommerce_sales.csv")
    DATA_PATH_JSON: str = os.path.join(BASE_DIR, "data", "ecommerce_sales.json")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")
    CHARTS_DIR: str = os.path.join(OUTPUT_DIR, "charts")
    
    # Export file paths
    COVARIANCE_CSV: str = os.path.join(OUTPUT_DIR, "covariance_matrix.csv")
    CORRELATION_CSV: str = os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
    RESULTS_CSV: str = os.path.join(OUTPUT_DIR, "correlation_results.csv")
    REPORT_TXT: str = os.path.join(OUTPUT_DIR, "relationship_report.txt")
    
    # Statistical analysis hyperparameters
    ALPHA: float = 0.05
    STRONG_THRESHOLD: float = 0.70
    MULTICOLLINEARITY_THRESHOLD: float = 0.75
    DIVERGENCE_THRESHOLD: float = 0.20
    TOP_N: int = 5
