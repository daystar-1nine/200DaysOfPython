import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def save_plot(output_path, dpi=150):
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

def plot_actual_vs_predicted(y_true, y_pred, output_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Actual vs Predicted')
    save_plot(output_path)

def plot_residual_plot(y_pred, residuals, output_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Sales')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    save_plot(output_path)

def plot_residual_distribution(residuals, output_path):
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    save_plot(output_path)

def plot_degree_vs_train_error(degrees, train_errors, output_path):
    plt.figure(figsize=(8, 6))
    plt.plot(degrees, train_errors, marker='o', label='Train Error')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Error (RMSE)')
    plt.title('Degree vs Train Error')
    plt.legend()
    save_plot(output_path)

def plot_degree_vs_test_error(degrees, test_errors, output_path):
    plt.figure(figsize=(8, 6))
    plt.plot(degrees, test_errors, marker='o', color='red', label='Test Error')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Error (RMSE)')
    plt.title('Degree vs Test Error')
    plt.legend()
    save_plot(output_path)

def plot_alpha_vs_error(alphas, cv_errors, output_path):
    plt.figure(figsize=(8, 6))
    plt.plot(alphas, cv_errors, marker='o')
    plt.xscale('log')
    plt.xlabel('Alpha (Log Scale)')
    plt.ylabel('CV RMSE')
    plt.title('Alpha vs Cross-Validation Error')
    save_plot(output_path)

def plot_model_comparison(comparison_df, output_path):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Test_RMSE', data=comparison_df)
    plt.xticks(rotation=45)
    plt.title('Model Comparison (Test RMSE)')
    save_plot(output_path)

def plot_correlation_heatmap(df, numeric_cols, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    save_plot(output_path)

def plot_coefficient_comparison(coef_comp_df, output_path):
    if coef_comp_df.empty: return
    plt.figure(figsize=(12, 6))
    coef_comp_df.set_index('Feature').plot(kind='bar', figsize=(12, 6))
    plt.title('Coefficient Comparison')
    plt.ylabel('Coefficient Value')
    save_plot(output_path)

def plot_advertising_sales(df, output_path):
    if 'Total_Ad_Spend' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='Total_Ad_Spend', y='Sales', data=df)
        plt.title('Total Advertising vs Sales')
        save_plot(output_path)

def plot_monthly_sales(df, output_path):
    if 'Month' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.lineplot(x='Month', y='Sales', data=df, estimator='mean', errorbar=None)
        plt.title('Average Monthly Sales')
        save_plot(output_path)

def plot_complexity_vs_error(degrees, train_errors, test_errors, output_path):
    plt.figure(figsize=(8, 6))
    plt.plot(degrees, train_errors, marker='o', label='Train Error')
    plt.plot(degrees, test_errors, marker='s', label='Test Error')
    plt.xlabel('Model Complexity (Degree)')
    plt.ylabel('Error (RMSE)')
    plt.title('Complexity vs Error (Overfitting Analysis)')
    plt.legend()
    save_plot(output_path)
