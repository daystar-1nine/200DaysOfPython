
import pandas as pd
import numpy as np

def calculate_business_cost(y_true, y_prob, threshold, fp_cost, fn_cost, tp_benefit=0, tn_benefit=0):
    y_pred = (y_prob >= threshold).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    cost = (fp * fp_cost) + (fn * fn_cost)
    benefit = (tp * tp_benefit) + (tn * tn_benefit)
    
    net_cost = cost - benefit
    return net_cost, tp, fp, fn, tn
    
def find_optimal_threshold(y_true, y_prob, fp_cost, fn_cost):
    thresholds = np.linspace(0, 1, 100)
    costs = []
    
    for t in thresholds:
        cost, _, _, _, _ = calculate_business_cost(y_true, y_prob, t, fp_cost, fn_cost)
        costs.append(cost)
        
    optimal_idx = np.argmin(costs)
    return thresholds[optimal_idx], costs[optimal_idx], thresholds, costs
