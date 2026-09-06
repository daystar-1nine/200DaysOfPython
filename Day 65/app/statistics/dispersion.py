"""
Dispersion statistical computation module.
"""

import pandas as pd
import numpy as np


def calculate_range(series: pd.Series) -> float:
    clean_s = series.dropna()
    if clean_s.empty:
        return np.nan
    return float(clean_s.max() - clean_s.min())


def calculate_variance(series: pd.Series, ddof: int = 1) -> float:
    clean_s = series.dropna()
    if len(clean_s) <= ddof:
        return np.nan
    return float(clean_s.var(ddof=ddof))


def calculate_std(series: pd.Series, ddof: int = 1) -> float:
    clean_s = series.dropna()
    if len(clean_s) <= ddof:
        return np.nan
    return float(clean_s.std(ddof=ddof))


def calculate_iqr(series: pd.Series) -> float:
    clean_s = series.dropna()
    if clean_s.empty:
        return np.nan
    q1 = float(clean_s.quantile(0.25))
    q3 = float(clean_s.quantile(0.75))
    return q3 - q1
