import numpy as np
from app.evaluation.calibration import compute_calibration

def test_compute_calibration():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    calib = compute_calibration(y_true, y_prob, n_bins=3)
    assert "prob_true" in calib
    assert "prob_pred" in calib
    assert "brier_score" in calib
    assert 0.0 <= calib["brier_score"] <= 1.0
