import os
import pandas as pd
from typing import List
from app.config import AppConfig

def generate_executive_report(
    model_comp: pd.DataFrame,
    best_params: dict,
    best_cv_score: float,
    threshold_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    optimal_t_f1: float,
    optimal_t_cost: float,
    insights: List[str],
    sample_explanation: dict,
    config: AppConfig
) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("DAY 78: CUSTOMER CHURN DECISION TREE ENGINE - EXECUTIVE REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("1. MODEL COMPARISON (DECISION TREE VARIANTS VS LOGISTIC BASELINE)")
    lines.append("-" * 80)
    lines.append(model_comp.to_string(index=False))
    lines.append("")
    
    lines.append("2. HYPERPARAMETER TUNING & REGULARIZATION (5-FOLD STRATIFIED CV)")
    lines.append("-" * 80)
    lines.append(f"Best CV F1-Score:     {best_cv_score:.4f}")
    lines.append("Optimal Hyperparameters:")
    for k, v in best_params.items():
        lines.append(f"  * {k}: {v}")
    lines.append("")
    
    lines.append("3. DECISION THRESHOLD & FINANCIAL COST OPTIMIZATION")
    lines.append("-" * 80)
    lines.append(f"Optimal Threshold for F1-Score:           {optimal_t_f1:.2f}")
    lines.append(f"Optimal Threshold for Minimal Cost:       {optimal_t_cost:.2f}")
    def_cost = cost_df.loc[cost_df['Threshold'] == 0.50, 'Total_Cost'].values[0] if (cost_df['Threshold'] == 0.50).any() else 0
    min_cost = cost_df['Total_Cost'].min()
    lines.append(f"Default Business Cost (at tau=0.50):       INR {def_cost:,.2f}")
    lines.append(f"Optimized Business Cost (at tau={optimal_t_cost:.2f}):   INR {min_cost:,.2f}")
    lines.append(f"Net Projected Financial Savings:          INR {def_cost - min_cost:,.2f}")
    lines.append("")
    
    lines.append("4. CUSTOMER-LEVEL DECISION PATH EXPLANATION")
    lines.append("-" * 80)
    lines.append(f"Customer ID:       {sample_explanation['Customer_ID']}")
    lines.append(f"Predicted Class:   {sample_explanation['Predicted_Class']}")
    lines.append(f"Churn Probability: {sample_explanation['Churn_Probability']:.4f}")
    lines.append("Inference Decision Path:")
    for step in sample_explanation['Decision_Path']:
        lines.append(f"  -> {step}")
    lines.append("")
    
    lines.append("5. AUTOMATED BUSINESS INSIGHTS")
    lines.append("-" * 80)
    for i, ins in enumerate(insights, 1):
        lines.append(f"{i}. {ins}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    content = "\n".join(lines)
    with open(os.path.join(config.OUTPUT_DIR, "model_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(content)
        
    return content
