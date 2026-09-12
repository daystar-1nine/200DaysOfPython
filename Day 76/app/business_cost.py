import pandas as pd
import numpy as np

def calculate_business_cost(fn_count: int, fp_count: int, fn_cost: float = 5000.0, fp_cost: float = 500.0) -> float:
    return (fn_count * fn_cost) + (fp_count * fp_cost)

def cost_curve_analysis(y_true, probabilities, thresholds, fn_cost: float = 5000.0, fp_cost: float = 500.0) -> pd.DataFrame:
    from app.threshold import apply_threshold
    
    results = []
    for t in thresholds:
        y_pred = apply_threshold(probabilities, t)
        
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        total_cost = calculate_business_cost(fn, fp, fn_cost, fp_cost)
        
        results.append({
            'Threshold': t,
            'FP': fp,
            'FN': fn,
            'Total_Cost': total_cost
        })
        
    return pd.DataFrame(results)
