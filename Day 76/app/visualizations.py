import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def plot_churn_distribution(df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Churn')
    plt.title('Churn Distribution')
    plt.savefig(output_path)
    plt.close()

def plot_age_vs_churn(df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Churn', y='Age')
    plt.title('Age vs Churn')
    plt.savefig(output_path)
    plt.close()

def plot_charges_vs_churn(df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Churn', y='Monthly_Charges')
    plt.title('Monthly Charges vs Churn')
    plt.savefig(output_path)
    plt.close()

def plot_tenure_vs_churn(df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Churn', y='Tenure_Months')
    plt.title('Tenure vs Churn')
    plt.savefig(output_path)
    plt.close()

def plot_contract_churn(df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Contract_Type', hue='Churn')
    plt.title('Contract Type vs Churn')
    plt.savefig(output_path)
    plt.close()

def plot_internet_churn(df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Internet_Service', hue='Churn')
    plt.title('Internet Service vs Churn')
    plt.savefig(output_path)
    plt.close()

def plot_support_calls_churn(df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Churn', y='Support_Calls')
    plt.title('Support Calls vs Churn')
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix_heatmap(cm_matrix: np.ndarray, output_path: str):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: float, output_path: str):
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_threshold_metrics(threshold_df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_df['Threshold'], threshold_df['Precision'], label='Precision')
    plt.plot(threshold_df['Threshold'], threshold_df['Recall'], label='Recall')
    plt.plot(threshold_df['Threshold'], threshold_df['F1'], label='F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Threshold')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_probability_distribution(probabilities: np.ndarray, y_true: np.ndarray, output_path: str):
    plt.figure(figsize=(10, 6))
    sns.histplot(x=probabilities, hue=y_true, bins=50, kde=True)
    plt.xlabel('Predicted Probability')
    plt.title('Probability Distribution by Class')
    plt.savefig(output_path)
    plt.close()
