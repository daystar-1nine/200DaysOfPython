import os
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'

def plot_churn_distribution(df: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {0: '#2ecc71', 1: '#e74c3c'}
    sns.countplot(data=df, x='Churn', hue='Churn', palette=palette, legend=False, ax=ax, edgecolor='black')
    ax.set_title('Customer Target Distribution (Stay vs Churn)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Churn Status (0 = Stay, 1 = Churn)', fontsize=11)
    ax.set_ylabel('Customer Count', fontsize=11)
    ax.set_xticklabels(['Stay (0)', 'Churn (1)'])
    
    total = len(df)
    for p in ax.patches:
        h = p.get_height()
        pct = (h / total) * 100.0
        ax.annotate(f'{h} ({pct:.1f}%)', (p.get_x() + p.get_width() / 2., h),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_churn_by_category(df: pd.DataFrame, cat_col: str, title: str, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    rates = df.groupby(cat_col)['Churn'].mean().reset_index()
    rates['Churn_Rate_Pct'] = rates['Churn'] * 100.0
    sns.barplot(data=rates, x=cat_col, y='Churn_Rate_Pct', hue=cat_col, palette='viridis', legend=False, ax=ax, edgecolor='black')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Churn Rate (%)', fontsize=11)
    ax.set_xlabel(cat_col.replace('_', ' '), fontsize=11)
    ax.set_ylim([0, max(rates['Churn_Rate_Pct']) * 1.25])
    
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f'{h:.1f}%', (p.get_x() + p.get_width() / 2., h),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_tenure_distribution(df: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.kdeplot(data=df[df['Churn'] == 0]['Tenure_Months'], label='Stay (0)', fill=True, color='#2ecc71', alpha=0.4, ax=ax)
    sns.kdeplot(data=df[df['Churn'] == 1]['Tenure_Months'], label='Churn (1)', fill=True, color='#e74c3c', alpha=0.4, ax=ax)
    ax.set_title('Customer Tenure Distribution by Churn Status', fontsize=13, fontweight='bold')
    ax.set_xlabel('Tenure (Months)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_charges_by_churn(df: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {0: '#2ecc71', 1: '#e74c3c'}
    sns.boxplot(data=df, x='Churn', y='Monthly_Charges', hue='Churn', palette=palette, legend=False, ax=ax)
    ax.set_title('Monthly Charges Distribution by Churn Status', fontsize=13, fontweight='bold')
    ax.set_xticklabels(['Stay (0)', 'Churn (1)'])
    ax.set_xlabel('Customer Status', fontsize=11)
    ax.set_ylabel('Monthly Charges (INR / Currency)', fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_cv_metric_comparison(cv_results: Dict[str, Dict[str, Dict[str, float]]], metric_key: str, title: str, output_path: str):
    models = list(cv_results.keys())
    means = [cv_results[m][metric_key]['test_mean'] for m in models]
    stds = [cv_results[m][metric_key]['test_std'] for m in models]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(models, means, yerr=stds, capsize=5, color='#3498db', edgecolor='black', alpha=0.85)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel(metric_key.replace('_', ' ').title(), fontsize=11)
    ax.set_ylim([0, 1.1])
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', (bar.get_x() + bar.get_width() / 2., h),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_test_metrics_grouped(df_comp: pd.DataFrame, output_path: str):
    metrics = ['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_ROC_AUC']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    
    plot_data = []
    for _, row in df_comp.iterrows():
        for m, l in zip(metrics, labels):
            plot_data.append({'Model': row['Model'], 'Metric': l, 'Score': row[m]})
    df_p = pd.DataFrame(plot_data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_p, x='Metric', y='Score', hue='Model', palette='Set2', ax=ax)
    ax.set_title('Test Set Model Performance Benchmark', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Metric Score', fontsize=11)
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_train_vs_val(cv_results: Dict[str, Dict[str, Dict[str, float]]], output_path: str):
    models = list(cv_results.keys())
    train_scores = [cv_results[m]['roc_auc']['train_mean'] for m in models]
    val_scores = [cv_results[m]['roc_auc']['test_mean'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, train_scores, width, label='Train ROC-AUC', color='#2980b9', edgecolor='black')
    ax.bar(x + width/2, val_scores, width, label='Validation ROC-AUC', color='#27ae60', edgecolor='black')
    ax.set_title('Train vs Validation ROC-AUC (Overfitting Diagnostic)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('ROC-AUC Score', fontsize=11)
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_confusion_matrix(cm_dict: Dict[str, int], model_name: str, output_path: str):
    matrix = np.array([[cm_dict['TN'], cm_dict['FP']], [cm_dict['FN'], cm_dict['TP']]])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(f'{model_name} Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('Actual Label', fontsize=10)
    ax.set_xticklabels(['Stay (0)', 'Churn (1)'])
    ax.set_yticklabels(['Stay (0)', 'Churn (1)'])
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_roc_curves(roc_data: Dict[str, Dict[str, Any]], output_path: str):
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = ['#95a5a6', '#e74c3c', '#e67e22', '#2ecc71']
    
    for idx, (name, d) in enumerate(roc_data.items()):
        ax.plot(d['fpr'], d['tpr'], label=f"{name} (AUC = {d['auc']:.3f})",
                linewidth=2.2, color=palette[idx % len(palette)])
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Guess (0.500)')
    ax.set_title('ROC Curves — All Models Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_pr_curves(pr_data: Dict[str, Dict[str, Any]], base_rate: float, output_path: str):
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = ['#95a5a6', '#e74c3c', '#e67e22', '#2ecc71']
    
    for idx, (name, d) in enumerate(pr_data.items()):
        ax.plot(d['recall'], d['precision'], label=f"{name} (AP = {d['ap']:.3f})",
                linewidth=2.2, color=palette[idx % len(palette)])
    ax.axhline(base_rate, color='gray', linestyle='--', alpha=0.7, label=f'Baseline Rate ({base_rate:.2f})')
    ax.set_title('Precision-Recall Curves — All Models Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='lower left')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_threshold_curves(df_thresh: pd.DataFrame, opt_t: float, output_path: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(df_thresh['Threshold'], df_thresh['Precision'], label='Precision', color='#3498db', linewidth=2)
    ax.plot(df_thresh['Threshold'], df_thresh['Recall'], label='Recall', color='#e74c3c', linewidth=2)
    ax.plot(df_thresh['Threshold'], df_thresh['F1'], label='F1 Score', color='#2ecc71', linewidth=2.5)
    ax.axvline(opt_t, color='purple', linestyle='--', label=f'Optimal F1 Threshold ({opt_t:.2f})')
    ax.set_title('Threshold Tuning: Precision, Recall & F1 Trade-off', fontsize=13, fontweight='bold')
    ax.set_xlabel('Decision Threshold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.legend(loc='lower left')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_cost_curve(cost_df: pd.DataFrame, opt_t: float, min_cost: float, default_cost: float, output_path: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(cost_df['Threshold'], cost_df['Total_Cost'], color='#8e44ad', linewidth=2.5, label='Total Business Cost')
    ax.axvline(opt_t, color='green', linestyle='--', label=f'Optimal Cost Threshold ({opt_t:.2f})')
    ax.axvline(0.50, color='red', linestyle=':', label='Default 0.50 Threshold')
    ax.scatter([opt_t], [min_cost], color='green', s=120, zorder=5, label=f'Min Cost (INR {min_cost:,.0f})')
    ax.scatter([0.50], [default_cost], color='red', s=100, zorder=5, label=f'Default Cost (INR {default_cost:,.0f})')
    ax.fill_between(cost_df['Threshold'], cost_df['Total_Cost'], default_cost,
                    where=(cost_df['Total_Cost'] <= default_cost), color='lightgreen', alpha=0.3, label='Cost Savings Zone')
    ax.set_title('Business Cost vs Classification Threshold', fontsize=13, fontweight='bold')
    ax.set_xlabel('Decision Threshold', fontsize=11)
    ax.set_ylabel('Total Cost (INR)', fontsize=11)
    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_feature_importance_bar(df_imp: pd.DataFrame, title: str, output_path: str, top_n: int = 12):
    df_top = df_imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    xerr = df_top['Std'] if 'Std' in df_top.columns else None
    ax.barh(df_top['Feature'], df_top['Importance'], xerr=xerr, color='#2980b9', capsize=3, edgecolor='black')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_permutation_bar(df_perm: pd.DataFrame, output_path: str, top_n: int = 12):
    df_top = df_perm.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df_top['Feature'], df_top['Mean_Importance'], xerr=df_top['Std_Importance'],
            color='#d35400', capsize=3, edgecolor='black')
    ax.set_title(f'Top {top_n} Features by Test Permutation Importance', fontsize=13, fontweight='bold')
    ax.set_xlabel('Mean ROC-AUC Degradation on Shuffling', fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_calibration_curves(calib_data: Dict[str, Dict[str, Any]], output_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    colors = ['#e74c3c', '#e67e22', '#2ecc71']
    
    for idx, (name, d) in enumerate(calib_data.items()):
        ax.plot(d['pred'], d['true'], marker='o', label=f"{name} (Brier = {d['brier']:.3f})",
                color=colors[idx % len(colors)], linewidth=2)
    ax.set_title('Probability Calibration Reliability Curves', fontsize=13, fontweight='bold')
    ax.set_xlabel('Mean Predicted Probability', fontsize=11)
    ax.set_ylabel('Fraction of Positives (Empirical Churn)', fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
