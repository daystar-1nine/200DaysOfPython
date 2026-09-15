import numpy as np
from app.business_cost import calculate_business_cost, evaluate_cost_curve, find_optimal_cost_threshold

def test_calculate_business_cost():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    cost = calculate_business_cost(y_true, y_pred, fp_cost=300.0, fn_cost=5000.0)
    assert cost["FP"] == 1
    assert cost["FN"] == 1
    assert cost["Total_Cost"] == 5300.0

def test_evaluate_cost_curve():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    df_c = evaluate_cost_curve(y_true, y_prob, fp_cost=300.0, fn_cost=5000.0, thresholds=[0.3, 0.5, 0.7])
    assert len(df_c) == 3
    assert "Total_Cost" in df_c.columns

def test_find_optimal_cost_threshold():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    df_c = evaluate_cost_curve(y_true, y_prob, fp_cost=300.0, fn_cost=5000.0, thresholds=[0.3, 0.5, 0.7])
    best_t = find_optimal_cost_threshold(df_c)
    assert best_t in [0.3, 0.5, 0.7]
