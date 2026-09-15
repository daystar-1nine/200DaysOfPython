"""Practice 4: Business Misclassification Cost & ROI Calculation."""
import numpy as np

def compute_roi(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fp: float = 300.0,
    cost_fn: float = 2000.0,
    retention_saving_tp: float = 1700.0
):
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    total_cost = fp * cost_fp + fn * cost_fn
    total_value_saved = tp * retention_saving_tp
    net_benefit = total_value_saved - total_cost
    
    return {
        'FP': int(fp), 'FN': int(fn), 'TP': int(tp), 'TN': int(tn),
        'Total_Cost': total_cost,
        'Value_Saved': total_value_saved,
        'Net_Benefit': net_benefit
    }

if __name__ == '__main__':
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    
    res = compute_roi(y_true, y_pred, cost_fp=300.0, cost_fn=2000.0, retention_saving_tp=1700.0)
    print("Cost & Benefit Breakdown:", res)
    assert res['TP'] == 4
    assert res['FP'] == 1
    assert res['FN'] == 1
    assert res['Total_Cost'] == 1 * 300.0 + 1 * 2000.0
    print("Practice 4 passed!")
