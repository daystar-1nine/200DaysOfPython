import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def compute_calibration(y_true, y_prob, n_bins: int = 10) -> dict:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    brier = float(brier_score_loss(y_true, y_prob))
    return {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "brier_score": brier
    }
