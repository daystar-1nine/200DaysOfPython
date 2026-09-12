import os
import pandas as pd
from typing import List
from app.config import AppConfig

def generate_text_report(
    binary_comp: pd.DataFrame,
    multi_comp: pd.DataFrame,
    threshold_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    optimal_t_f1: float,
    optimal_t_cost: float,
    insights: List[str],
    config: AppConfig
):
    lines = []
    lines.append("=" * 80)
    lines.append("DAY 77: INTELLIGENT CUSTOMER RISK CLASSIFICATION ENGINE — EXECUTIVE REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("1. BINARY CLASSIFICATION (CHURN PREDICTION) MODEL COMPARISON")
    lines.append("-" * 80)
    lines.append(binary_comp.to_string(index=False))
    lines.append("")
    
    lines.append("2. MULTI-CLASS CLASSIFICATION (CUSTOMER RISK LEVEL) MODEL COMPARISON")
    lines.append("-" * 80)
    lines.append(multi_comp.to_string(index=False))
    lines.append("")
    
    lines.append("3. DECISION THRESHOLD & BUSINESS COST OPTIMIZATION")
    lines.append("-" * 80)
    lines.append(f"Optimal Threshold for F1-Score:           {optimal_t_f1:.2f}")
    lines.append(f"Optimal Threshold for Minimal Cost:       {optimal_t_cost:.2f}")
    def_cost = cost_df.loc[cost_df['Threshold'] == 0.50, 'Total_Cost'].values[0] if (cost_df['Threshold'] == 0.50).any() else 0
    min_cost = cost_df['Total_Cost'].min()
    lines.append(f"Default Business Cost (at tau=0.50):       INR {def_cost:,.2f}")
    lines.append(f"Optimized Business Cost (at tau={optimal_t_cost:.2f}):   INR {min_cost:,.2f}")
    lines.append(f"Net Projected Savings:                    INR {def_cost - min_cost:,.2f}")
    lines.append("")
    
    lines.append("4. AUTOMATED BUSINESS INSIGHTS")
    lines.append("-" * 80)
    for i, ins in enumerate(insights, 1):
        lines.append(f"{i}. {ins}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    content = "\n".join(lines)
    with open(config.CLASSIFICATION_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(content)
        
    return content
