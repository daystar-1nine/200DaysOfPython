import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def calculate_business_cost(y_true, y_pred, fp_cost: float = 300.0, fn_cost: float = 5000.0) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total_cost = (fp * fp_cost) + (fn * fn_cost)
    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "FP_Cost": float(fp * fp_cost),
        "FN_Cost": float(fn * fn_cost),
        "Total_Cost": float(total_cost)
    }

def evaluate_cost_curve(y_true, y_prob, fp_cost: float = 300.0, fn_cost: float = 5000.0, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
        
    records = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        cost_info = calculate_business_cost(y_true, preds, fp_cost=fp_cost, fn_cost=fn_cost)
        records.append({
            "Threshold": round(float(t), 2),
            "FP": cost_info["FP"],
            "FN": cost_info["FN"],
            "TP": cost_info["TP"],
            "TN": cost_info["TN"],
            "FP_Cost": cost_info["FP_Cost"],
            "FN_Cost": cost_info["FN_Cost"],
            "Total_Cost": cost_info["Total_Cost"]
        })
    return pd.DataFrame(records)

def find_optimal_cost_threshold(cost_df: pd.DataFrame) -> float:
    best_row = cost_df.loc[cost_df["Total_Cost"].idxmin()]
    return float(best_row["Threshold"])
