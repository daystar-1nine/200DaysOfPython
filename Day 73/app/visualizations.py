"""
Day 73 - Visualization Engine
Generates 8 publication-quality diagnostic charts using headless matplotlib Agg backend.
"""
import os
from typing import Dict, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def generate_all_charts(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    scenarios_df: pd.DataFrame,
    output_dir: str
):
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    y_test_pred = model.predict(X_test)
    residuals_test = y_test - y_test_pred
    
    # 1. Raw Scatter: Advertising vs Sales
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[feature_col], df[target_col], color="#1f77b4", alpha=0.5, edgecolors="none")
    ax.set_title("Raw Data: Advertising Spend vs. Sales Revenue", fontsize=12, fontweight="bold")
    ax.set_xlabel("Advertising Spend (Rs.)")
    ax.set_ylabel("Sales Revenue (Rs.)")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "advertising_sales.png"), dpi=300)
    plt.close()
    
    # 2. Regression Line Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_test, y_test, color="#1f77b4", alpha=0.5, label="Test Observations")
    x_line = np.linspace(df[feature_col].min(), df[feature_col].max(), 200).reshape(-1, 1)
    y_line = model.predict(x_line)
    ax.plot(x_line, y_line, color="#d62728", linewidth=2.5, label=f"Fitted OLS: {model.get_equation()}")
    ax.set_title("Fitted Simple Linear Regression Line", fontsize=12, fontweight="bold")
    ax.set_xlabel("Advertising Spend (Rs.)")
    ax.set_ylabel("Sales Revenue (Rs.)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "regression_line.png"), dpi=300)
    plt.close()
    
    # 3. Actual vs Predicted Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_test_pred, color="#2ca02c", alpha=0.6, edgecolors="black", linewidth=0.5)
    lims = [min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())]
    ax.plot(lims, lims, color="#d62728", linestyle="--", linewidth=2, label="Perfect 1:1 Parity")
    ax.set_title("Actual vs. Predicted Sales (Test Set)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual Sales (Rs.)")
    ax.set_ylabel("Predicted Sales (Rs.)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "actual_predicted.png"), dpi=300)
    plt.close()
    
    # 4. Residual Plot (Fitted vs Residuals)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test_pred, residuals_test, color="#9467bd", alpha=0.6)
    ax.axhline(0, color="#d62728", linestyle="--", linewidth=2)
    ax.set_title("Residual Plot: Predicted Values vs. Residuals", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Sales (Rs.)")
    ax.set_ylabel("Residual (Actual - Predicted) (Rs.)")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "residual_plot.png"), dpi=300)
    plt.close()
    
    # 5. Residual Distribution (Histogram & KDE)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals_test, kde=True, color="#ff7f0e", ax=ax, bins=15)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_title("Test Residual Error Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Residual Error (Rs.)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "residual_distribution.png"), dpi=300)
    plt.close()
    
    # 6. Train vs Test Performance Metrics Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_names = ["MAE (Rs.)", "RMSE (Rs.)", "R2 Score"]
    train_vals = [train_metrics["mae"], train_metrics["rmse"], train_metrics["r2"]]
    test_vals = [test_metrics["mae"], test_metrics["rmse"], test_metrics["r2"]]
    
    # For visualization, plot R2 on secondary axis or scale
    x_indices = np.arange(2)
    width = 0.35
    ax.bar(x_indices - width/2, [train_vals[0], train_vals[1]], width, label="Train Set", color="#1f77b4")
    ax.bar(x_indices + width/2, [test_vals[0], test_vals[1]], width, label="Test Set", color="#ff7f0e")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(["MAE (Rs.)", "RMSE (Rs.)"])
    ax.set_title(f"Train vs. Test Error (Train R2: {train_vals[2]:.3f} | Test R2: {test_vals[2]:.3f})", fontsize=12, fontweight="bold")
    ax.set_ylabel("Error Magnitude (Rs.)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "train_test_metrics.png"), dpi=300)
    plt.close()
    
    # 7. Scenario Planning Predictions Chart
    fig, ax = plt.subplots(figsize=(9, 5.5))
    normal_scenarios = scenarios_df[~scenarios_df["Is_Extrapolation"]]
    extrap_scenarios = scenarios_df[scenarios_df["Is_Extrapolation"]]
    
    ax.bar(normal_scenarios["Advertising_Spend"] / 1000, normal_scenarios["Predicted_Sales"] / 1000,
           color="#2ca02c", width=6.0, alpha=0.8, label="Supported Interpolation")
    if not extrap_scenarios.empty:
        ax.bar(extrap_scenarios["Advertising_Spend"] / 1000, extrap_scenarios["Predicted_Sales"] / 1000,
               color="#d62728", width=6.0, alpha=0.8, label="Extrapolation Risk")
               
    ax.set_title("Forecasted Sales Across Advertising Spending Budgets", fontsize=12, fontweight="bold")
    ax.set_xlabel("Advertising Budget (Thousands Rs.)")
    ax.set_ylabel("Forecasted Sales (Thousands Rs.)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "prediction_scenarios.png"), dpi=300)
    plt.close()
    
    # 8. Correlation Visualization
    fig, ax = plt.subplots(figsize=(7, 6))
    corr_mat = df[[feature_col, target_col]].corr()
    sns.heatmap(corr_mat, annot=True, fmt=".3f", cmap="Blues", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Pearson Correlation Heatmap", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "correlation.png"), dpi=300)
    plt.close()
    
    print(f"Generated 8 publication charts in {charts_dir}")
