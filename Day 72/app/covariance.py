"""
Day 72 — Covariance Calculation Engine
Calculates pairwise sample and population covariance and covariance matrices.
"""
import numpy as np
import pandas as pd

def compute_sample_covariance(x: np.ndarray, y: np.ndarray, ddof: int = 1) -> float:
    n = len(x)
    if n != len(y):
        raise ValueError("Inputs x and y must have equal length.")
    if n <= ddof:
        raise ValueError(f"Sample size {n} must exceed ddof {ddof}.")
        
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    cov_val = np.sum((x - mean_x) * (y - mean_y)) / (n - ddof)
    return float(cov_val)

def compute_covariance_matrix(df: pd.DataFrame, ddof: int = 1) -> pd.DataFrame:
    return df.cov(ddof=ddof)
