import numpy as np
import pandas as pd
from app.business_cost import compute_business_cost, sweep_cost_thresholds, find_minimum_cost_threshold

def test_compute_business_cost_formula():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])  # 1 TP, 1 FP, 1 FN, 1 TN
    res = compute_business_cost(y_true, y_pred, cost_fp=300.0, cost_fn=2000.0)
    assert res['FP_Count'] == 1
    assert res['FN_Count'] == 1
    assert res['FP_Cost'] == 300.0
    assert res['FN_Cost'] == 2000.0
    assert res['Total_Cost'] == 2300.0
    assert res['Avg_Cost_Per_Customer'] == 2300.0 / 4

def test_compute_business_cost_perfect_prediction():
    y_true = np.array([1, 0, 1, 0])
    res = compute_business_cost(y_true, y_true, cost_fp=300.0, cost_fn=2000.0)
    assert res['Total_Cost'] == 0.0

def test_sweep_cost_thresholds_length():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.7, 0.2])
    df_cost = sweep_cost_thresholds(y_true, y_prob, cost_fp=300.0, cost_fn=2000.0)
    assert len(df_cost) == 91
    assert 'Total_Cost' in df_cost.columns

def test_find_minimum_cost_threshold():
    df_cost = pd.DataFrame({
        'Threshold': [0.2, 0.4, 0.6, 0.8],
        'Total_Cost': [5000.0, 3200.0, 4100.0, 7000.0]
    })
    min_t, min_cost = find_minimum_cost_threshold(df_cost)
    assert min_t == 0.4
    assert min_cost == 3200.0

def test_cost_positive():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    df_cost = sweep_cost_thresholds(y_true, y_prob, cost_fp=300.0, cost_fn=2000.0)
    assert (df_cost['Total_Cost'] >= 0.0).all()
