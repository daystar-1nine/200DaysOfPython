"""Stratified Cross Validation runner."""
from typing import Dict, Tuple, Any
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

def run_stratified_cv(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, n_splits: int = 5, random_state: int = 42) -> Tuple[float, float]:
    """Runs n-fold Stratified CV inside the pipeline to strictly prevent data leakage."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring="f1")
    return float(np.mean(scores)), float(np.std(scores))
