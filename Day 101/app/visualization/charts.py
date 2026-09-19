"""Generates all 12 required NLP visualizations."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

def save_all_visualizations(
    df: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_comparison_df: pd.DataFrame,
    top_spam: list,
    top_ham: list,
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

    # 4. Top Words Overall
    all_words = " ".join(df["text"]).lower().split()
    top_words = pd.Series(all_words).value_counts().head(15)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_words.values, y=top_words.index, hue=top_words.index, palette="Blues_r", legend=False)
    plt.title("4. Top 15 Words Overall")
    plt.savefig(output_dir / "4_top_words_overall.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Top Spam Words / Features
    if top_spam:
        words, weights = zip(*top_spam[:12])
        plt.figure(figsize=(8, 5))
        sns.barplot(x=list(weights), y=list(words), hue=list(words), palette="Reds_r", legend=False)
        plt.title("5. Top Predictive Spam Features (Logistic Coefs)")
        plt.savefig(output_dir / "5_top_spam_words.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 6. Top Ham Words / Features
    if top_ham:
        words, weights = zip(*top_ham[:12])
        plt.figure(figsize=(8, 5))
        sns.barplot(x=[abs(w) for w in weights], y=list(words), hue=list(words), palette="Greens_r", legend=False)
        plt.title("6. Top Predictive Ham Features (Logistic Coefs |Magnitude|)")
        plt.savefig(output_dir / "6_top_ham_words.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 7. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham (0)", "Spam (1)"], yticklabels=["Ham (0)", "Spam (1)"])
    plt.title("7. Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(output_dir / "7_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 8. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2980b9", lw=2, label="ROC Curve")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("8. Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.savefig(output_dir / "8_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 9. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#8e44ad", lw=2, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("9. Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(output_dir / "9_precision_recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 10. Model Comparison
    if not model_comparison_df.empty:
        plt.figure(figsize=(9, 5))
        melted = model_comparison_df.melt(id_vars=["model"], value_vars=["accuracy", "precision", "recall", "f1"], var_name="metric", value_name="score")
        sns.barplot(data=melted, x="metric", y="score", hue="model", palette="Set2")
        plt.ylim(0, 1.05)
        plt.title("10. Model Performance Comparison")
        plt.savefig(output_dir / "10_model_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 11. Prediction Confidence Distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(y_proba, bins=25, kde=True, color="#16a085")
    plt.title("11. Prediction Confidence Distribution (Spam Probability)")
    plt.xlabel("Predicted P(Spam)")
    plt.savefig(output_dir / "11_prediction_confidence_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 12. Misclassified Examples Analysis
    errors_mask = y_test != y_pred
    error_counts = pd.Series(["Correct" if not e else "Misclassified" for e in errors_mask]).value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=error_counts.index, y=error_counts.values, hue=error_counts.index, palette=["#27ae60", "#c0392b"], legend=False)
    plt.title("12. Misclassified vs Correct Predictions")
    plt.ylabel("Number of Messages")
    plt.savefig(output_dir / "12_misclassified_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print("Saved all 12 visualizations successfully.")
