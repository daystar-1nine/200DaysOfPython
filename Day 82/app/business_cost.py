
import numpy as np

def calculate_total_cost(y_true, y_pred, fp_cost=300, fn_cost=2000):
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return fp, fn, (fp * fp_cost) + (fn * fn_cost)
    
def evaluate_thresholds(y_true, probabilities, thresholds, fp_cost=300, fn_cost=2000):
    results = []
    for t in thresholds:
        y_pred = (probabilities >= t).astype(int)
        fp, fn, cost = calculate_total_cost(y_true, y_pred, fp_cost, fn_cost)
        results.append({
            'threshold': t,
            'fp': fp,
            'fn': fn,
            'cost': cost
        })
    return results
