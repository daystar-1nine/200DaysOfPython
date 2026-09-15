import numpy as np
import pandas as pd
from app.threshold import sweep_thresholds, find_optimal_threshold

def test_sweep_thresholds_shape():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.3, 0.85, 0.6])
    df_t = sweep_thresholds(y_true, y_prob)
    assert isinstance(df_t, pd.DataFrame)
    assert len(df_t) == 91
    assert 'Threshold' in df_t.columns
    assert 'F1_Score' in df_t.columns

def test_threshold_metrics_bounded():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.2, 0.7, 0.3])
    df_t = sweep_thresholds(y_true, y_prob)
    assert (df_t['Precision'] >= 0.0).all() and (df_t['Precision'] <= 1.0).all()
    assert (df_t['Recall'] >= 0.0).all() and (df_t['Recall'] <= 1.0).all()
    assert (df_t['F1_Score'] >= 0.0).all() and (df_t['F1_Score'] <= 1.0).all()

def test_find_optimal_threshold():
    df_t = pd.DataFrame({
        'Threshold': [0.2, 0.4, 0.6, 0.8],
        'F1_Score': [0.5, 0.85, 0.75, 0.4]
    })
    opt_t, opt_val = find_optimal_threshold(df_t, 'F1_Score')
    assert opt_t == 0.4
    assert opt_val == 0.85

def test_recall_at_low_vs_high_threshold():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.3, 0.85, 0.6])
    df_t = sweep_thresholds(y_true, y_prob)
    low_rec = df_t.iloc[0]['Recall']
    high_rec = df_t.iloc[-1]['Recall']
    assert low_rec >= high_rec
