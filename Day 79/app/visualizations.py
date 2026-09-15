import os
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, average_precision_score

sns.set_theme(style='whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'

def plot_confusion_matrices(
    cms: Dict[str, np.ndarray], 
    output_path: str
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    names = list(cms.keys())
    
    for idx, name in enumerate(names):
        cm = cms[name]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
        axes[idx].set_title(f'{name} Confusion Matrix', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
        axes[idx].set_ylabel('Actual Label', fontsize=10)
        axes[idx].set_xticklabels(['Retained (0)', 'Churn (1)'])
        axes[idx].set_yticklabels(['Retained (0)', 'Churn (1)'])
        
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_roc_curves(
    roc_data: Dict[str, Dict[str, Any]], 
    output_path: str
):
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']
    
    for idx, (name, data) in enumerate(roc_data.items()):
        ax.plot(data['fpr'], data['tpr'], label=f"{name} (AUC = {data['auc']:.3f})",
                linewidth=2.2, color=palette[idx % len(palette)])
        
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance (AUC = 0.500)')
    ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=11)
    ax.legend(loc='lower right', frameon=True)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_pr_curves(
    pr_data: Dict[str, Dict[str, Any]], 
    base_rate: float,
    output_path: str
):
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']
    
    for idx, (name, data) in enumerate(pr_data.items()):
        ax.plot(data['recall'], data['precision'], label=f"{name} (AP = {data['ap']:.3f})",
                linewidth=2.2, color=palette[idx % len(palette)])
        
    ax.axhline(base_rate, color='gray', linestyle='--', alpha=0.7, label=f'Baseline Rate ({base_rate:.2f})')
    ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.legend(loc='lower left', frameon=True)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_model_benchmarks(
    metrics_dict: Dict[str, Dict[str, float]], 
    output_path: str
):
    records = []
    target_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    metric_labels = {'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall',
                     'f1_score': 'F1 Score', 'roc_auc': 'ROC AUC'}
    
    for model_name, m_dict in metrics_dict.items():
        for m in target_metrics:
            records.append({
                'Model': model_name,
                'Metric': metric_labels[m],
                'Score': m_dict.get(m, 0.0)
            })
    df_plot = pd.DataFrame(records)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_plot, x='Metric', y='Score', hue='Model', palette='Set2', ax=ax)
    ax.set_title('Model Performance Metrics Benchmark', fontsize=14, fontweight='bold')
    ax.set_ylim([0.0, 1.05])
    ax.set_ylabel('Score', fontsize=11)
    ax.legend(loc='lower right', frameon=True)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0.02:
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 2),
                        textcoords='offset points')
            
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_mdi_importance(
    df_imp: pd.DataFrame, 
    output_path: str,
    top_n: int = 12
):
    df_top = df_imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(df_top['Feature'], df_top['Importance'], xerr=df_top['Std'],
            color='#3498db', alpha=0.85, capsize=4, edgecolor='black')
    ax.set_title(f'Top {top_n} Features by Mean Decrease in Impurity (MDI / Gini)', fontsize=14, fontweight='bold')
    ax.set_xlabel('MDI Feature Importance (+/- std across trees)', fontsize=11)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_permutation_importance(
    df_perm: pd.DataFrame, 
    output_path: str,
    top_n: int = 12
):
    df_top = df_perm.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(df_top['Feature'], df_top['Mean_Importance'], xerr=df_top['Std_Importance'],
            color='#e67e22', alpha=0.85, capsize=4, edgecolor='black')
    ax.set_title(f'Top {top_n} Features by Permutation Importance (Test Set)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mean F1 Decrease on Feature Permutation', fontsize=11)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_mdi_vs_permutation(
    df_mdi: pd.DataFrame, 
    df_perm: pd.DataFrame, 
    output_path: str,
    top_n: int = 10
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    mdi_top = df_mdi.head(top_n).iloc[::-1]
    axes[0].barh(mdi_top['Feature'], mdi_top['Importance'], color='#2980b9')
    axes[0].set_title(f'MDI Impurity Importance (Top {top_n})', fontweight='bold')
    axes[0].set_xlabel('Importance Score')
    
    perm_top = df_perm.head(top_n).iloc[::-1]
    axes[1].barh(perm_top['Feature'], perm_top['Mean_Importance'], color='#d35400')
    axes[1].set_title(f'Permutation Importance (Top {top_n})', fontweight='bold')
    axes[1].set_xlabel('F1 Drop on Permutation')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_threshold_curves(
    df_thresh: pd.DataFrame, 
    opt_t: float, 
    opt_f1: float,
    output_path: str
):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_thresh['Threshold'], df_thresh['Precision'], label='Precision', color='#3498db', linewidth=2)
    ax.plot(df_thresh['Threshold'], df_thresh['Recall'], label='Recall', color='#e74c3c', linewidth=2)
    ax.plot(df_thresh['Threshold'], df_thresh['F1_Score'], label='F1 Score', color='#2ecc71', linewidth=2.5)
    ax.axvline(opt_t, color='purple', linestyle='--', label=f'Optimal F1 Threshold ({opt_t:.2f})')
    ax.scatter([opt_t], [opt_f1], color='purple', s=100, zorder=5)
    
    ax.set_title('Threshold Sweep: Precision, Recall & F1 Trade-off', fontsize=14, fontweight='bold')
    ax.set_xlabel('Decision Threshold', fontsize=11)
    ax.set_ylabel('Metric Score', fontsize=11)
    ax.set_xlim([0.05, 0.95])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='lower left', frameon=True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_cost_curve(
    cost_df: pd.DataFrame, 
    opt_t: float, 
    min_cost: float,
    default_cost: float,
    output_path: str
):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cost_df['Threshold'], cost_df['Total_Cost'], color='#8e44ad', linewidth=2.5, label='Total Business Cost')
    ax.axvline(opt_t, color='green', linestyle='--', label=f'Optimal Cost Threshold ({opt_t:.2f})')
    ax.axvline(0.50, color='red', linestyle=':', label='Default Threshold (0.50)')
    ax.scatter([opt_t], [min_cost], color='green', s=120, zorder=5, label=f'Min Cost (INR {min_cost:,.0f})')
    ax.scatter([0.50], [default_cost], color='red', s=100, zorder=5, label=f'Default Cost (INR {default_cost:,.0f})')
    
    ax.fill_between(cost_df['Threshold'], cost_df['Total_Cost'], default_cost, 
                    where=(cost_df['Total_Cost'] <= default_cost), color='lightgreen', alpha=0.3, label='Value Creation Zone')
    
    ax.set_title('Total Business Cost vs Classification Threshold', fontsize=14, fontweight='bold')
    ax.set_xlabel('Classification Decision Threshold', fontsize=11)
    ax.set_ylabel('Total Cost (INR)', fontsize=11)
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_oob_error_vs_trees(
    tree_counts: List[int], 
    oob_errors: List[float], 
    output_path: str
):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tree_counts, oob_errors, marker='o', color='#2980b9', linewidth=2, markersize=6)
    ax.set_title('Out-of-Bag (OOB) Error vs Number of Estimators', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Trees (n_estimators)', fontsize=11)
    ax.set_ylabel('OOB Error Rate (1 - OOB Accuracy)', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_tree_variance_reduction(
    individual_tree_probs: np.ndarray, 
    ensemble_probs: np.ndarray, 
    output_path: str
):
    fig, ax = plt.subplots(figsize=(10, 6))
    # Select sample of 5 individual trees
    for i in range(min(5, individual_tree_probs.shape[1])):
        sns.kdeplot(individual_tree_probs[:, i], ax=ax, alpha=0.35, linestyle=':', label=f'Tree #{i+1}')
        
    sns.kdeplot(ensemble_probs, ax=ax, color='black', linewidth=3, label='Random Forest Ensemble (Mean)')
    ax.set_title('Variance Reduction: Individual Trees vs Random Forest Ensemble', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Probability of Churn', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_cv_boxplots(
    cv_records: pd.DataFrame, 
    output_path: str
):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=cv_records, x='Model', y='F1_Score', hue='Model', legend=False, palette='Pastel1', ax=ax)
    sns.stripplot(data=cv_records, x='Model', y='F1_Score', color='black', size=6, jitter=0.2, ax=ax)
    ax.set_title('5-Fold Stratified Cross-Validation F1 Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1 Score across Folds', fontsize=11)
    ax.set_xlabel('Model Architecture', fontsize=11)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_risk_distribution(
    df_risk: pd.DataFrame, 
    output_path: str
):
    fig, ax = plt.subplots(figsize=(10, 6))
    tier_order = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
    palette = {'Low Risk': '#2ecc71', 'Medium Risk': '#f1c40f', 'High Risk': '#e67e22', 'Critical Risk': '#e74c3c'}
    
    sns.countplot(data=df_risk, x='Risk_Tier', hue='Risk_Tier', legend=False, order=tier_order, palette=palette, ax=ax, edgecolor='black')
    ax.set_title('Customer Distribution across Churn Risk Tiers', fontsize=14, fontweight='bold')
    ax.set_xlabel('Risk Classification Tier', fontsize=11)
    ax.set_ylabel('Customer Count', fontsize=11)
    
    total = len(df_risk)
    for p in ax.patches:
        height = p.get_height()
        pct = 100.0 * height / total
        ax.annotate(f'{height} ({pct:.1f}%)',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                    textcoords='offset points')
                    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_cost_savings_waterfall(
    unmanaged_cost: float, 
    default_cost: float, 
    optimal_cost: float, 
    output_path: str
):
    fig, ax = plt.subplots(figsize=(9, 6))
    categories = ['Unmanaged Baseline', 'Default Threshold (0.50)', 'Optimal Threshold (Cost-Tuned)']
    costs = [unmanaged_cost, default_cost, optimal_cost]
    colors = ['#c0392b', '#e67e22', '#27ae60']
    
    bars = ax.bar(categories, costs, color=colors, edgecolor='black', width=0.55)
    ax.set_title('Misclassification Cost Reduction & Business Impact', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Cost (INR)', fontsize=11)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'INR {height:,.0f}',
                    (bar.get_x() + bar.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 4),
                    textcoords='offset points')
                    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
