import pandas as pd
import json
from typing import List

def generate_report(metrics: dict, cm_dict: dict, optimal_threshold: float, insights: List[str], output_path: str):
    with open(output_path, 'w') as f:
        f.write("=== CUSTOMER CHURN PREDICTION REPORT ===\n\n")
        
        f.write("--- METRICS ---\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n")
        
        f.write("--- CONFUSION MATRIX ---\n")
        f.write(f"True Positives: {cm_dict['TP']}\n")
        f.write(f"True Negatives: {cm_dict['TN']}\n")
        f.write(f"False Positives: {cm_dict['FP']}\n")
        f.write(f"False Negatives: {cm_dict['FN']}\n")
        f.write("\n")
        
        f.write(f"--- THRESHOLD ---\n")
        f.write(f"Optimal Threshold (F1): {optimal_threshold:.2f}\n")
        f.write("\n")
        
        f.write("--- BUSINESS INSIGHTS ---\n")
        for i, insight in enumerate(insights, 1):
            f.write(f"{i}. {insight}\n")
