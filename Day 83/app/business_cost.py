
import numpy as np

def calculate_business_cost(y_true, y_prob, threshold, fp_cost=300, fn_cost=2000):
    y_pred = (y_prob >= threshold).astype(int)
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return (fp * fp_cost) + (fn * fn_cost)
