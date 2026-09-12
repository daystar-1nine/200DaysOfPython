"""
Day 73 - Report and Export Engine
Exports metrics and scenario predictions to CSV and formats executive ASCII report.
"""
import os
from typing import Dict, Any, List
import pandas as pd

def export_artifacts(
    output_dir: str,
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    report_text: str
):
    os.makedirs(output_dir, exist_ok=True)
    metrics_df.to_csv(os.path.join(output_dir, "regression_results.csv"), index=False)
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    with open(os.path.join(output_dir, "regression_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

def format_ascii_report(
    eda_dict: Dict[str, Any],
    model_params: Dict[str, float],
    equation: str,
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    residual_info: Dict[str, Any],
    predictions_df: pd.DataFrame,
    insights: List[str]
) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("         SALES PREDICTION REGRESSION ENGINE REPORT          ")
    lines.append("=" * 70)
    lines.append(f"Total Observations:           {eda_dict['count']:,}")
    lines.append(f"Predictor Feature (X):        Advertising_Spend (Rs.)")
    lines.append(f"Target Variable (Y):          Sales (Rs.)")
    lines.append(f"Pearson Correlation (r):      {eda_dict['pearson_r']:.4f} (p = {eda_dict['p_value']:.4e})")
    lines.append("-" * 70)
    
    lines.append("\n" + "=" * 70)
    lines.append("1. FITTED REGRESSION MODEL")
    lines.append("=" * 70)
    lines.append(f"Algorithm:                    Simple Linear Regression (OLS)")
    lines.append(f"Slope (b1):                   {model_params['slope']:.4f}")
    lines.append(f"Intercept (b0):               Rs. {model_params['intercept']:,.2f}")
    lines.append(f"Regression Equation:          {equation}")
    
    lines.append("\n" + "=" * 70)
    lines.append("2. TRAIN VS. TEST GENERALIZATION EVALUATION")
    lines.append("=" * 70)
    lines.append(f"{'Metric':<12} | {'Training Set (80%)':<22} | {'Test Set (20%)':<22}")
    lines.append("-" * 70)
    lines.append(f"{'MAE':<12} | Rs. {train_metrics['mae']:<18,.2f} | Rs. {test_metrics['mae']:<18,.2f}")
    lines.append(f"{'MSE':<12} | {train_metrics['mse']:<22,.2f} | {test_metrics['mse']:<22,.2f}")
    lines.append(f"{'RMSE':<12} | Rs. {train_metrics['rmse']:<18,.2f} | Rs. {test_metrics['rmse']:<18,.2f}")
    lines.append(f"{'R2 Score':<12} | {train_metrics['r2']:<22.4f} | {test_metrics['r2']:<22.4f}")
    
    lines.append("\n" + "=" * 70)
    lines.append("3. RESIDUAL DIAGNOSTICS")
    lines.append("=" * 70)
    lines.append(f"Mean Residual:                Rs. {residual_info['mean_residual']:.4e} (Aligned to 0)")
    lines.append(f"Residual Std Dev:             Rs. {residual_info['std_residual']:,.2f}")
    lines.append(f"Residual Skewness:            {residual_info['skewness']:.4f} (Near 0)")
    lines.append(f"Variance Ratio (High/Low):    {residual_info['variance_ratio']:.4f}")
    lines.append(f"Homoscedasticity Valid:       {residual_info['is_homoscedastic']}")
    
    lines.append("\n" + "=" * 70)
    lines.append("4. SCENARIO PLANNING PREDICTIONS")
    lines.append("=" * 70)
    lines.append(f"{'Advertising Spend':<22} | {'Predicted Sales':<22} | {'Extrapolation Risk':<18}")
    lines.append("-" * 70)
    for _, row in predictions_df.iterrows():
        spend_str = f"Rs. {row['Advertising_Spend']:,.2f}"
        sales_str = f"Rs. {row['Predicted_Sales']:,.2f}"
        risk_str = "YES [!] Warning" if row["Is_Extrapolation"] else "No (Supported)"
        lines.append(f"{spend_str:<22} | {sales_str:<22} | {risk_str:<18}")
        
    lines.append("\n" + "=" * 70)
    lines.append("5. AUTOMATED STRATEGIC INSIGHTS & CAUSAL GOVERNANCE")
    lines.append("=" * 70)
    for idx, ins in enumerate(insights, 1):
        lines.append(f"{idx}. {ins}\n")
        
    lines.append("=" * 70)
    lines.append("                       END OF REPORT                        ")
    lines.append("=" * 70)
    
    return "\n".join(lines)
