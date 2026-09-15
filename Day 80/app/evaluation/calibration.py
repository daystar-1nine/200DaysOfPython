from typing import Tuple
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def calculate_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute calibration curve and Brier score."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    brier = float(brier_score_loss(y_true, y_prob))
    return prob_true, prob_pred, brier
