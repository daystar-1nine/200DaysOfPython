"""Generates all 15 required NLP analytical charts."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

def generate_all_charts(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    feature_exp_df: pd.DataFrame,
    trained_pipes: dict,
    top_spam: pd.DataFrame,
    top_ham: pd.DataFrame,
    y_test: np.ndarray,
    errors_df: pd.DataFrame,
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Class Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="label", hue="label", palette={"ham": "#2ecc71", "spam": "#e74c3c"}, legend=False)
    plt.title("1. Class Distribution (Ham vs Spam)")
    plt.savefig(output_dir / "1_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Message Length Distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x="char_length", hue="label", bins=40, kde=True, palette={"ham": "#2ecc71", "spam": "#e74c3c"})
    plt.title("2. Message Length Distribution (Characters)")
    plt.savefig(output_dir / "2_message_length_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Word Count Distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x="word_count", hue="label", bins=30, kde=True, palette={"ham": "#2ecc71", "spam": "#e74c3c"})
    plt.title("3. Word Count Distribution")
    plt.savefig(output_dir / "3_word_count_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Top Vocabulary Terms
    all_words = " ".join(df["text"]).lower().split()
    top_words = pd.Series(all_words).value_counts().head(15)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_words.values, y=top_words.index, hue=top_words.index, palette="Blues_r", legend=False)
    plt.title("4. Top 15 Vocabulary Terms Overall")
    plt.savefig(output_dir / "4_top_vocabulary_terms.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Top Spam-Associated Terms (Coefficients)
    if not top_spam.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=top_spam.head(12), x="coefficient", y="feature", hue="feature", palette="Reds_r", legend=False)
        plt.title("5. Top 12 Spam-Associated Features (Model Coefficients)")
        plt.savefig(output_dir / "5_top_spam_terms.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 6. Top Ham-Associated Terms
    if not top_ham.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=np.abs(top_ham.head(12)["coefficient"]), y=top_ham.head(12)["feature"], hue=top_ham.head(12)["feature"], palette="Greens_r", legend=False)
        plt.title("6. Top 12 Ham-Associated Features (|Coefficients|)")
        plt.savefig(output_dir / "6_top_ham_terms.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 7. Model F1 Comparison
    plt.figure(figsize=(9, 4))
    benchmark_df["Model_Config"] = benchmark_df["Model"] + " (" + benchmark_df["Features"] + ")"
    sns.barplot(data=benchmark_df, x="F1", y="Model_Config", hue="Model_Config", palette="viridis", legend=False)
    plt.xlim(0, 1.05)
    plt.title("7. Model F1 Score Comparison")
    plt.savefig(output_dir / "7_model_f1_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 8. Model Precision Comparison
    plt.figure(figsize=(9, 4))
    sns.barplot(data=benchmark_df, x="Precision", y="Model_Config", hue="Model_Config", palette="crest", legend=False)
    plt.xlim(0, 1.05)
    plt.title("8. Model Precision Comparison")
    plt.savefig(output_dir / "8_model_precision_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 9. Model Recall Comparison
    plt.figure(figsize=(9, 4))
    sns.barplot(data=benchmark_df, x="Recall", y="Model_Config", hue="Model_Config", palette="magma", legend=False)
    plt.xlim(0, 1.05)
    plt.title("9. Model Recall Comparison")
    plt.savefig(output_dir / "9_model_recall_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 10. Confusion Matrix (Best Model: Logistic Regression Word TF-IDF)
    best_pipe = trained_pipes.get("Logistic Regression (Word TF-IDF)")
    if best_pipe:
        cm = confusion_matrix(y_test, best_pipe["predictions"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
        plt.title("10. Confusion Matrix (Logistic Regression Word TF-IDF)")
        plt.ylabel("Actual Label")
        plt.xlabel("Predicted Label")
        plt.savefig(output_dir / "10_confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 11. ROC Curves (Multi-model)
    plt.figure(figsize=(7, 6))
    for name, data in trained_pipes.items():
        fpr, tpr, _ = roc_curve(y_test, data["scores"])
        auc = data["metrics"].get("roc_auc", 0.0)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.title("11. Multi-Model ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(output_dir / "11_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 12. Precision-Recall Curves
    plt.figure(figsize=(7, 6))
    for name, data in trained_pipes.items():
        prec, rec, _ = precision_recall_curve(y_test, data["scores"])
        ap = data["metrics"].get("avg_precision", 0.0)
        plt.plot(rec, prec, lw=2, label=f"{name} (AP={ap:.3f})")
    plt.title("12. Multi-Model Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.savefig(output_dir / "12_precision_recall_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 13. Feature Count Comparison
    plt.figure(figsize=(8, 4))
    sns.barplot(data=feature_exp_df, x="vocab_size", y="experiment", hue="experiment", palette="Blues_d", legend=False)
    plt.title("13. Feature Vocabulary Size Comparison Across Representations")
    plt.xlabel("Number of Features")
    plt.savefig(output_dir / "13_feature_count_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 14. CV Mean +/- Variation
    plt.figure(figsize=(9, 4))
    plt.errorbar(
        x=benchmark_df["CV Mean"],
        y=range(len(benchmark_df)),
        xerr=benchmark_df["CV Std"],
        fmt="o",
        color="#2980b9",
        ecolor="#e74c3c",
        elinewidth=2,
        capsize=4
    )
    plt.yticks(range(len(benchmark_df)), benchmark_df["Model_Config"])
    plt.xlim(0.85, 1.02)
    plt.title("14. 5-Fold Stratified Cross-Validation F1 (Mean ± Std)")
    plt.xlabel("Cross-Validation F1 Score")
    plt.savefig(output_dir / "14_cv_mean_variation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 15. Error Distribution
    plt.figure(figsize=(6, 4))
    if not errors_df.empty and "error_type" in errors_df.columns:
        sns.countplot(data=errors_df, x="error_type", hue="error_type", palette="Set1", legend=False)
        plt.title("15. Error Distribution (FP vs FN Across Models)")
    else:
        plt.text(0.5, 0.5, "Zero Prediction Errors Observed!", ha="center", va="center", fontsize=12, color="green")
        plt.title("15. Error Distribution (Zero Errors)")
    plt.savefig(output_dir / "15_error_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print("Generated all 15 visualizations successfully.")
