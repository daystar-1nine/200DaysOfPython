"""
Distribution shape, skewness, and kurtosis module.
"""

import pandas as pd
import numpy as np


def calculate_skewness(series: pd.Series) -> float:
    clean_s = series.dropna()
    if len(clean_s) < 3:
        return 0.0
    return float(clean_s.skew())


def calculate_kurtosis(series: pd.Series) -> float:
    clean_s = series.dropna()
    if len(clean_s) < 4:
        return 0.0
    return float(clean_s.kurt())


def classify_distribution(skewness: float, kurtosis: float) -> dict[str, str]:
    if abs(skewness) < 0.5:
        skew_class = "Approximately Symmetric"
    elif skewness >= 0.5:
        skew_class = "Positively Skewed (Right-Tail Heavy)"
    else:
        skew_class = "Negatively Skewed (Left-Tail Heavy)"
        
    if kurtosis > 1.0:
        kurt_class = "Leptokurtic (Heavy Tails, High Outlier Risk)"
    elif kurtosis < -1.0:
        kurt_class = "Platykurtic (Light Tails, Low Outlier Risk)"
    else:
        kurt_class = "Mesokurtic (Normal-like Tail Weight)"
        
    return {
        "skewness_classification": skew_class,
        "kurtosis_classification": kurt_class
    }
