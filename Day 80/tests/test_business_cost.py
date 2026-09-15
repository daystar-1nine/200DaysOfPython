import numpy as np
import pandas as pd
from app.business_cost import compute_total_cost, sweep_cost_curve, find_optimal_cost_threshold

def test_compute_total_cost_formula():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    res = compute_total_cost(y_true, y_pred, cost_fp=300.0, cost_fn=2000.0)
    assert res['FP'] == 1
    assert res['FN'] == 1
    assert res['Total_Cost'] == 2300.0
    assert res['Avg_Cost_Per_Customer'] == 2300.0 / 4

def test_zero_cost_on_perfect():
    y_true = np.array([1, 0, 1, 0])
    res = compute_total_cost(y_true, y_true, cost_fp=300.0, cost_fn=2000.0)
    assert res['Total_Cost'] == 0.0

def test_sweep_cost_curve_size():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.7, 0.2])
    df = sweep_cost_curve(y_true, y_prob, cost_fp=300.0, cost_fn=2000.0)
    assert len(df) == 91
    assert 'Total_Cost' in df.columns

def test_find_optimal_cost_threshold():
    df = pd.DataFrame({
        'Threshold': [0.2, 0.4, 0.6, 0.8],
        'Total_Cost': [6000.0, 3100.0, 4500.0, 7000.0]
    })
    opt_t, min_c = find_optimal_cost_threshold(df)
    assert opt_t == 0.4
    assert min_c == 3100.0

def test_cost_non_negative():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    df = sweep_cost_curve(y_true, y_prob)
    assert (df['Total_Cost'] >= 0.0).all()
