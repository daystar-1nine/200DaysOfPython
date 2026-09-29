"""
Controlled benchmark comparing TF-IDF + LR, GRU + Attention, Transformer, and BERT models.
"""
import time
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from app.evaluation.metrics import compute_metrics


def run_tfidf_baseline(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int]
) -> Dict[str, Any]:
    """
    Fits TF-IDF + Logistic Regression baseline on identical train/test splits.
    """
    start_t = time.time()
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, train_labels)
    fit_time = time.time() - start_t

    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = compute_metrics(test_labels, preds, probs)
    metrics["model"] = "TF-IDF + Logistic Regression"
    metrics["parameters"] = X_train.shape[1] + 1
    metrics["training_time_seconds"] = round(fit_time, 4)
    return metrics


def build_benchmark_table(
    tfidf_metrics: Dict[str, Any],
    frozen_bert_metrics: Dict[str, Any],
    finetuned_bert_metrics: Dict[str, Any],
    output_path: Path = None
) -> pd.DataFrame:
    """
    Builds the complete cross-architecture comparison table:
    1. TF-IDF + Logistic Regression
    2. GRU + Attention (Day 109)
    3. Transformer (Day 110)
    4. BERT Frozen Backbone (Day 111 Experiment A)
    5. BERT Fine-Tuned (Day 111 Experiment B)
    """
    records = [
        {
            "Model": "TF-IDF + Logistic Regression",
            "Architecture": "Bag of Words + Linear",
            "Parameters": tfidf_metrics.get("parameters", 1001),
            "Trainable Params": tfidf_metrics.get("parameters", 1001),
            "Training Time (s)": tfidf_metrics.get("training_time_seconds", 0.05),
            "Accuracy": tfidf_metrics["accuracy"],
            "Precision": tfidf_metrics["precision"],
            "Recall": tfidf_metrics["recall"],
            "F1": tfidf_metrics["f1"],
            "ROC-AUC": tfidf_metrics["roc_auc"]
        },
        {
            "Model": "GRU + Attention (Day 109)",
            "Architecture": "Recurrent + Additive Attention",
            "Parameters": 157825,
            "Trainable Params": 157825,
            "Training Time (s)": 1.59,
            "Accuracy": 0.9850,
            "Precision": 0.9620,
            "Recall": 0.9380,
            "F1": 0.9498,
            "ROC-AUC": 0.9912
        },
        {
            "Model": "Mini Transformer (Day 110)",
            "Architecture": "Self-Attention + Positional Encoding",
            "Parameters": 315393,
            "Trainable Params": 315393,
            "Training Time (s)": 5.74,
            "Accuracy": 0.9880,
            "Precision": 0.9710,
            "Recall": 0.9450,
            "F1": 0.9578,
            "ROC-AUC": 0.9945
        },
        {
            "Model": "BERT Frozen Backbone",
            "Architecture": "Pretrained Transformer (Head Only)",
            "Parameters": frozen_bert_metrics.get("total_params", 4394241),
            "Trainable Params": frozen_bert_metrics.get("trainable_params", 8321),
            "Training Time (s)": frozen_bert_metrics.get("training_time_seconds", 4.2),
            "Accuracy": frozen_bert_metrics["accuracy"],
            "Precision": frozen_bert_metrics["precision"],
            "Recall": frozen_bert_metrics["recall"],
            "F1": frozen_bert_metrics["f1"],
            "ROC-AUC": frozen_bert_metrics["roc_auc"]
        },
        {
            "Model": "BERT Fine-Tuned",
            "Architecture": "Pretrained Transformer (End-to-End)",
            "Parameters": finetuned_bert_metrics.get("total_params", 4394241),
            "Trainable Params": finetuned_bert_metrics.get("trainable_params", 4394241),
            "Training Time (s)": finetuned_bert_metrics.get("training_time_seconds", 8.5),
            "Accuracy": finetuned_bert_metrics["accuracy"],
            "Precision": finetuned_bert_metrics["precision"],
            "Recall": finetuned_bert_metrics["recall"],
            "F1": finetuned_bert_metrics["f1"],
            "ROC-AUC": finetuned_bert_metrics["roc_auc"]
        }
    ]

    df = pd.DataFrame(records)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Benchmark table saved to: {output_path}")

    return df
