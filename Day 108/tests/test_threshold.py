import pytest
import numpy as np
import pandas as pd
from app.evaluation.threshold import analyze_thresholds

def test_analyze_thresholds_structure(tmp_path):
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.85, 0.9, 0.35, 0.4, 0.15, 0.25])
    csv_out = tmp_path / "thresh.csv"
    
    df = analyze_thresholds(y_true, y_prob, model_name="GRU_Test", save_path=csv_out)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 9 # 0.10 to 0.90
    assert "threshold" in df.columns
    assert "precision" in df.columns
    assert "recall" in df.columns
    assert "false_positives" in df.columns
    assert "false_negatives" in df.columns
    assert csv_out.exists()

def test_analyze_thresholds_monotonicity():
    # As threshold increases, false positives should be non-increasing
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.85, 0.9, 0.35, 0.4, 0.15, 0.25])
    
    df = analyze_thresholds(y_true, y_prob, model_name="GRU_Test")
    fps = df["false_positives"].tolist()
    for i in range(len(fps) - 1):
        assert fps[i] >= fps[i+1]
