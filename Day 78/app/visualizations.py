import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.tree import plot_tree
from app.config import AppConfig

def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 14
    })

def plot_churn_distribution(df: pd.DataFrame, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Churn"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, palette=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title("Customer Churn Distribution (0=Retained, 1=Churned)")
    ax.set_xlabel("Churn Status")
    ax.set_ylabel("Customer Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, f"{v} ({v/len(df)*100:.1f}%)", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "churn_distribution.png"), dpi=150)
    plt.close()

def plot_contract_churn(df: pd.DataFrame, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ct = pd.crosstab(df["Contract_Type"], df["Churn"], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, color=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title("Observed Churn Rate by Contract Type (%)")
    ax.set_ylabel("Percentage")
    ax.legend(["Retained", "Churned"])
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "contract_churn.png"), dpi=150)
    plt.close()

def plot_internet_churn(df: pd.DataFrame, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ct = pd.crosstab(df["Internet_Service"], df["Churn"], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, color=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title("Observed Churn Rate by Internet Service (%)")
    ax.set_ylabel("Percentage")
    ax.legend(["Retained", "Churned"])
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "internet_churn.png"), dpi=150)
    plt.close()

def plot_charges_churn(df: pd.DataFrame, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="Churn", y="Monthly_Charges", hue="Churn", legend=False, palette=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title("Monthly Charges vs Churn Status")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Retained (0)", "Churned (1)"])
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "charges_churn.png"), dpi=150)
    plt.close()

def plot_tenure_churn(df: pd.DataFrame, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.violinplot(data=df, x="Churn", y="Tenure_Months", hue="Churn", legend=False, palette=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title("Tenure Months vs Churn Status")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Retained (0)", "Churned (1)"])
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "tenure_churn.png"), dpi=150)
    plt.close()

def plot_decision_tree_graph(pipeline, feature_names: list, config: AppConfig, max_depth=3):
    setup_style()
    fig, ax = plt.subplots(figsize=(18, 10))
    clf = pipeline.named_steps["classifier"]
    plot_tree(
        clf,
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=["Retained", "Churned"],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax
    )
    ax.set_title(f"Decision Tree Architecture (Pruned Visualization to Depth {max_depth})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "decision_tree.png"), dpi=200)
    plt.close()

def plot_feature_importance_bar(feat_df: pd.DataFrame, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    top = feat_df.head(10)
    sns.barplot(data=top, x="Importance", y="Feature", hue="Feature", legend=False, palette="viridis", ax=ax)
    ax.set_title("Top 10 Feature Importances (MDI / Gini Reduction)")
    ax.set_xlabel("Mean Decrease in Impurity")
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()

def plot_permutation_importance_bar(perm_df: pd.DataFrame, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    top = perm_df.head(10)
    ax.barh(top["Feature"], top["Importance_Mean"], xerr=top["Importance_Std"], color="#3498db", capsize=3)
    ax.invert_yaxis()
    ax.set_title("Top 10 Permutation Importances (F1 Score Decrease)")
    ax.set_xlabel("Mean Score Drop")
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "permutation_importance.png"), dpi=150)
    plt.close()

def plot_confusion_matrix_heatmap(cm_data: dict, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_data["dataframe"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Decision Tree Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

def plot_roc_curve_chart(roc_data: dict, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(roc_data["fpr"], roc_data["tpr"], color="#2980b9", lw=2, label=f"ROC Curve (AUC = {roc_data['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "roc_curve.png"), dpi=150)
    plt.close()

def plot_pr_curve_chart(pr_data: dict, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(pr_data["recall"], pr_data["precision"], color="#27ae60", lw=2, label=f"PR Curve (AP = {pr_data['average_precision']:.3f})")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "precision_recall_curve.png"), dpi=150)
    plt.close()

def plot_depth_vs_score_chart(depth_df: pd.DataFrame, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depth_df["Depth"], depth_df["Train_F1"], marker="o", color="#e74c3c", label="Train F1 (Overfitting Zone)")
    ax.plot(depth_df["Depth"], depth_df["Test_F1"], marker="s", color="#2980b9", label="Test F1 (Generalization)")
    ax.set_title("Decision Tree Depth vs F1-Score (Bias-Variance Tradeoff)")
    ax.set_xlabel("Maximum Tree Depth")
    ax.set_ylabel("F1 Score")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "depth_vs_score.png"), dpi=150)
    plt.close()
