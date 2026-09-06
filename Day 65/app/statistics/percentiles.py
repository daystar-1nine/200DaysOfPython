"""
Percentiles and quartiles computation module.
"""

import pandas as pd
import numpy as np


def calculate_percentile(series: pd.Series, q: float) -> float:
    clean_s = series.dropna()
    if clean_s.empty:
        return np.nan
    # q is in 0-100 range
    return float(np.percentile(clean_s, q))


def calculate_percentiles(series: pd.Series, percentiles: list[float] = None) -> dict[str, float]:
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        
    clean_s = series.dropna()
    if clean_s.empty:
        return {f"P{p:02d}": np.nan for p in percentiles}
        
    return {f"P{p:02d}": float(np.percentile(clean_s, p)) for p in percentiles}
