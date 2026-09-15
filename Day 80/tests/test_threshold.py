import numpy as np
import pandas as pd
from app.threshold import sweep_thresholds, find_best_f1_threshold

def test_sweep_thresholds_length():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.3, 0.85, 0.6])
    df = sweep_thresholds(y_true, y_prob)
    assert len(df) == 91
    assert 'Threshold' in df.columns
    assert 'F1' in df.columns

def test_precision_recall_tradeoff():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.3, 0.85, 0.6])
    df = sweep_thresholds(y_true, y_prob)
    assert df.iloc[0]['Recall'] >= df.iloc[-1]['Recall']

def test_find_best_f1_threshold():
    df = pd.DataFrame({
        'Threshold': [0.2, 0.4, 0.6, 0.8],
        'F1': [0.5, 0.85, 0.75, 0.4]
    })
    opt_t, opt_val = find_best_f1_threshold(df)
    assert opt_t == 0.4
    assert opt_val == 0.85

def test_threshold_metrics_bounded():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.2, 0.7, 0.3])
    df = sweep_thresholds(y_true, y_prob)
    assert (df['Precision'] >= 0.0).all() and (df['Precision'] <= 1.0).all()
    assert (df['Recall'] >= 0.0).all() and (df['Recall'] <= 1.0).all()
