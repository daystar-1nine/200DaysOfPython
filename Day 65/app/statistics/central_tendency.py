"""
Central tendency statistical computation module.
"""

import pandas as pd
import numpy as np


def calculate_mean(series: pd.Series) -> float:
    clean_s = series.dropna()
    if clean_s.empty:
        return np.nan
    return float(clean_s.mean())


def calculate_median(series: pd.Series) -> float:
    clean_s = series.dropna()
    if clean_s.empty:
        return np.nan
    return float(clean_s.median())


def calculate_mode(series: pd.Series) -> list:
    clean_s = series.dropna()
    if clean_s.empty:
        return []
    modes = clean_s.mode().tolist()
    return modes
