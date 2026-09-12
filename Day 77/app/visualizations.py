import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
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

def plot_risk_distribution(df: pd.DataFrame, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["Low", "Medium", "High"]
    counts = df["Risk_Level"].value_counts().reindex(order)
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, palette=["#3498db", "#f39c12", "#e74c3c"], ax=ax)
    ax.set_title("Customer Multi-Class Risk Level Distribution")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Customer Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, f"{v} ({v/len(df)*100:.1f}%)", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "risk_distribution.png"), dpi=150)
    plt.close()

def plot_age_vs_churn(df: pd.DataFrame, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(data=df, x="Age", hue="Churn", common_norm=False, fill=True, palette=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title("Age Distribution by Churn Status")
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "age_vs_churn.png"), dpi=150)
    plt.close()

def plot_charges_vs_churn(df: pd.DataFrame, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="Churn", y="Monthly_Charges", hue="Churn", legend=False, palette=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title("Monthly Charges vs Churn Status")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Retained (0)", "Churned (1)"])
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "charges_vs_churn.png"), dpi=150)
    plt.close()

def plot_tenure_vs_churn(df: pd.DataFrame, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.violinplot(data=df, x="Churn", y="Tenure_Months", hue="Churn", legend=False, palette=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title("Tenure Months vs Churn Status")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Retained (0)", "Churned (1)"])
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "tenure_vs_churn.png"), dpi=150)
    plt.close()

def plot_contract_churn(df: pd.DataFrame, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ct = pd.crosstab(df["Contract_Type"], df["Churn"], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, color=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title("Churn Rate by Contract Type (%)")
    ax.set_ylabel("Percentage")
    ax.legend(["Retained", "Churned"])
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "contract_churn.png"), dpi=150)
    plt.close()

def plot_binary_confusion_matrix(cm_data: dict, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_data["dataframe"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Binary Confusion Matrix (Churn)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "binary_confusion_matrix.png"), dpi=150)
    plt.close()

def plot_multiclass_confusion_matrix(cm_data: dict, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_data["normalized_df"], annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Normalized Multi-Class Confusion Matrix (Risk)")
    ax.set_xlabel("Predicted Risk Level")
    ax.set_ylabel("Actual Risk Level")
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "multiclass_confusion_matrix.png"), dpi=150)
    plt.close()

def plot_roc_curve(roc_binary: dict, roc_multi: dict, config: AppConfig):
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Binary ROC
    ax1.plot(roc_binary["fpr"], roc_binary["tpr"], color="#2980b9", lw=2, label=f"Binary ROC (AUC = {roc_binary['auc']:.3f})")
    ax1.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    ax1.scatter([roc_binary["fpr"][int(np.argmax(roc_binary['tpr'] - roc_binary['fpr']))]],
                [roc_binary["tpr"][int(np.argmax(roc_binary['tpr'] - roc_binary['fpr']))]],
                color="red", s=50, label=f"Youden J @ {roc_binary['optimal_threshold_youden']:.2f}")
    ax1.set_title("Binary ROC Curve")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")
    
    # Multiclass ROC (OvR)
    for cls_name, curve in roc_multi["roc_curves"].items():
        ax2.plot(curve["fpr"], curve["tpr"], lw=2, label=f"{cls_name} (AUC = {curve['auc']:.3f})")
    ax2.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    ax2.set_title(f"Multi-Class ROC OvR (Macro AUC = {roc_multi['macro_auc']:.3f})")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "roc_curve.png"), dpi=150)
    plt.close()

def plot_precision_recall_curve(pr_binary: dict, pr_multi: dict, config: AppConfig):
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Binary PR
    ax1.plot(pr_binary["recall"], pr_binary["precision"], color="#27ae60", lw=2, label=f"Binary PR (AP = {pr_binary['average_precision']:.3f})")
    ax1.set_title("Binary Precision-Recall Curve")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.legend(loc="lower left")
    
    # Multiclass PR
    for cls_name, curve in pr_multi["pr_curves"].items():
        ax2.plot(curve["recall"], curve["precision"], lw=2, label=f"{cls_name} (AP = {curve['average_precision']:.3f})")
    ax2.set_title(f"Multi-Class PR Curves (Macro AP = {pr_multi['macro_ap']:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "precision_recall_curve.png"), dpi=150)
    plt.close()

def plot_threshold_metrics(threshold_df: pd.DataFrame, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_df["Threshold"], threshold_df["Precision"], marker="o", color="#2980b9", label="Precision")
    ax.plot(threshold_df["Threshold"], threshold_df["Recall"], marker="s", color="#e67e22", label="Recall")
    ax.plot(threshold_df["Threshold"], threshold_df["Accuracy"], marker="^", color="#7f8c8d", label="Accuracy")
    ax.axvline(x=0.50, color="grey", linestyle="--", alpha=0.7, label="Default (0.50)")
    ax.set_title("Precision, Recall & Accuracy vs Decision Threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "threshold_metrics.png"), dpi=150)
    plt.close()

def plot_threshold_f1(threshold_df: pd.DataFrame, cost_df: pd.DataFrame, config: AppConfig):
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # F1 Curve
    best_f1_t = threshold_df.loc[threshold_df["F1_Score"].idxmax()]["Threshold"]
    max_f1 = threshold_df["F1_Score"].max()
    ax1.plot(threshold_df["Threshold"], threshold_df["F1_Score"], marker="o", color="#8e44ad", lw=2)
    ax1.axvline(x=best_f1_t, color="red", linestyle="--", label=f"Max F1 @ {best_f1_t:.2f} ({max_f1:.3f})")
    ax1.set_title("F1-Score vs Threshold")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("F1 Score")
    ax1.legend()
    
    # Cost Curve
    min_cost_t = cost_df.loc[cost_df["Total_Cost"].idxmin()]["Threshold"]
    min_cost = cost_df["Total_Cost"].min()
    ax2.plot(cost_df["Threshold"], cost_df["Total_Cost"], marker="o", color="#c0392b", lw=2)
    ax2.axvline(x=min_cost_t, color="green", linestyle="--", label=f"Min Cost @ {min_cost_t:.2f} (₹{min_cost:,.0f})")
    ax2.set_title("Estimated Business Cost vs Threshold")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Total Cost (INR)")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "threshold_f1.png"), dpi=150)
    plt.close()

def plot_probability_distribution(y_true, y_prob, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(x=y_prob, hue=y_true, bins=25, kde=True, palette=["#2ecc71", "#e74c3c"], ax=ax)
    ax.axvline(0.50, color="black", linestyle="--", label="Threshold = 0.50")
    ax.set_title("Predicted Churn Probability Distribution by True Class")
    ax.set_xlabel("Predicted Probability of Churn")
    ax.legend(["Retained (0)", "Churned (1)"])
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "probability_distribution.png"), dpi=150)
    plt.close()

def plot_calibration_curve(calib_info: dict, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    ax.plot(calib_info["prob_pred"], calib_info["prob_true"], "s-", color="#16a085", label=f"Logistic Regression (Brier={calib_info['brier_score']:.4f})")
    ax.set_title("Probability Calibration Curve")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "calibration_curve.png"), dpi=150)
    plt.close()

def plot_feature_importance(pipeline, feature_names: list, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    model = pipeline.named_steps["classifier"]
    
    if hasattr(model, "coef_"):
        coefs = model.coef_[0] if model.coef_.ndim > 1 and model.coef_.shape[0] == 1 else model.coef_[0]
        feat_df = pd.DataFrame({"Feature": feature_names[:len(coefs)], "Coefficient": coefs})
        feat_df = feat_df.sort_values(by="Coefficient", key=abs, ascending=False).head(12)
        sns.barplot(data=feat_df, x="Coefficient", y="Feature", palette="coolwarm", ax=ax)
        ax.set_title("Top Feature Coefficients (Logistic Regression)")
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_df = pd.DataFrame({"Feature": feature_names[:len(importances)], "Importance": importances})
        feat_df = feat_df.sort_values(by="Importance", ascending=False).head(12)
        sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis", ax=ax)
        ax.set_title("Top Feature Importances (Tree Model)")
        
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()

def plot_class_performance(multiclass_metrics: dict, config: AppConfig):
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    per_class = multiclass_metrics["per_class"]
    classes = list(per_class.keys())
    
    p = [per_class[c]["precision"] for c in classes]
    r = [per_class[c]["recall"] for c in classes]
    f = [per_class[c]["f1"] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    ax.bar(x - width, p, width, label="Precision", color="#3498db")
    ax.bar(x, r, width, label="Recall", color="#e67e22")
    ax.bar(x + width, f, width, label="F1-Score", color="#2ecc71")
    
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_title("Per-Class Multi-Class Classification Performance")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "class_performance.png"), dpi=150)
    plt.close()
