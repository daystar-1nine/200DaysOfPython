import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def _save(fig, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_actual_vs_predicted(y_true, y_pred, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.6)
    
    # 45-degree line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted Sales")
    _save(fig, output_path)

def plot_residuals_vs_fitted(y_pred, residuals, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel("Fitted Values (Predicted)")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    _save(fig, output_path)

def plot_residual_distribution(residuals, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title("Distribution of Residuals")
    _save(fig, output_path)

def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str], output_path: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
    ax.set_title("Correlation Heatmap")
    _save(fig, output_path)

def plot_feature_vs_sales(df: pd.DataFrame, feature: str, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(x=feature, y="Sales", data=df, ax=ax, scatter_kws={'alpha':0.5})
    ax.set_title(f"{feature} vs Sales")
    _save(fig, output_path)

def plot_coefficient_chart(coef_df: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Coefficient", y="Feature", data=coef_df, ax=ax, hue="Feature", palette="vlag", legend=False)
    ax.set_title("Model Coefficients")
    _save(fig, output_path)

def plot_sales_by_region(df: pd.DataFrame, output_path: str):
    if "Region" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x="Region", y="Sales", data=df, ax=ax)
        ax.set_title("Sales by Region")
        _save(fig, output_path)

def plot_monthly_sales(df: pd.DataFrame, output_path: str):
    if "Month" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        monthly_sales = df.groupby("Month")["Sales"].mean().reset_index()
        sns.lineplot(x="Month", y="Sales", data=monthly_sales, ax=ax, marker='o')
        ax.set_title("Average Sales by Month")
        ax.set_xticks(range(1, 13))
        _save(fig, output_path)

def plot_model_comparison(comparison_df: pd.DataFrame, output_path: str):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    sns.barplot(x="Model", y="R2", data=comparison_df, ax=ax1, color="blue", alpha=0.6, label="R2")
    ax1.set_ylabel("R2 Score", color="blue")
    
    ax2 = ax1.twinx()
    sns.lineplot(x="Model", y="RMSE", data=comparison_df, ax=ax2, color="red", marker='o', label="RMSE")
    ax2.set_ylabel("RMSE", color="red")
    
    plt.title("Model Comparison")
    _save(fig, output_path)

def plot_vif_chart(vif_df: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="VIF", y="Feature", data=vif_df, ax=ax, color="orange")
    ax.axvline(10, color='r', linestyle='--', label="Threshold (10)")
    ax.set_title("Variance Inflation Factor (VIF)")
    ax.legend()
    _save(fig, output_path)
