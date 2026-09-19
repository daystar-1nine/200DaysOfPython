"""Orchestration pipeline for Day 101 NLP project."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import config
from app.data.loader import load_sms_data
from app.data.cleaner import clean_dataset
from app.data.validator import validate_dataset
from app.features.tfidf import get_sklearn_tfidf
from app.models.baseline import get_baseline_model
from app.models.logistic import get_logistic_model
from app.models.naive_bayes import get_naive_bayes_model
from app.evaluation.metrics import calculate_classification_metrics
from app.evaluation.confusion_matrix import get_confusion_matrix_breakdown
from app.analysis.error_analysis import extract_prediction_errors, diagnose_errors
from app.analysis.insights import get_top_features_by_class
from app.visualization.charts import save_all_visualizations
from app.report import generate_markdown_report

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def run_pipeline():
    print("=== STARTING DAY 101 NLP PIPELINE ===")
    
    # 1. Load & clean data
    raw_df = load_sms_data(config.DATA_RAW)
    df = clean_dataset(raw_df)
    validation = validate_dataset(df)
    print(f"Data validated: {validation['total_rows']} records, Class distribution: {validation['class_distribution']}")
    
    df.to_csv(config.DATA_PROCESSED, index=False, encoding="utf-8")
    
    # 2. Train / Test Split
    X = df["text"].values
    y = np.where(df["label"] == "spam", 1, 0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 3. Model Pipelines
    baseline_pipe = Pipeline([
        ("tfidf", get_sklearn_tfidf()),
        ("classifier", get_baseline_model(random_state=config.RANDOM_STATE))
    ])
    
    logistic_pipe = Pipeline([
        ("tfidf", get_sklearn_tfidf(ngram_range=(1, 2))),
        ("classifier", get_logistic_model(C=1.0, random_state=config.RANDOM_STATE))
    ])
    
    nb_pipe = Pipeline([
        ("tfidf", get_sklearn_tfidf(ngram_range=(1, 2))),
        ("classifier", get_naive_bayes_model(alpha=0.5))
    ])
    
    # Train Models
    models = {
        "Dummy Baseline": baseline_pipe,
        "Logistic Regression (TF-IDF 1-2 Grams)": logistic_pipe,
        "Multinomial Naive Bayes": nb_pipe
    }
    
    model_records = []
    trained_pipelines = {}
    
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps["classifier"], "predict_proba") else None
        
        metrics = calculate_classification_metrics(y_test, preds, proba)
        metrics["model"] = name
        model_records.append(metrics)
        trained_pipelines[name] = (pipe, preds, proba)
        print(f"[{name}] Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
        
    comp_df = pd.DataFrame(model_records)[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"]]
    comp_df.to_csv(config.METRICS_PATH, index=False)
    
    # Primary Model Evaluation (Logistic Regression)
    primary_pipe, primary_preds, primary_proba = trained_pipelines["Logistic Regression (TF-IDF 1-2 Grams)"]
    primary_metrics = calculate_classification_metrics(y_test, primary_preds, primary_proba)
    cm_breakdown = get_confusion_matrix_breakdown(y_test, primary_preds)
    
    # Top features
    top_spam, top_ham = get_top_features_by_class(primary_pipe, top_n=15)
    
    # Error Analysis
    error_df = extract_prediction_errors(X_test, y_test, primary_preds, primary_proba)
    diagnostics = diagnose_errors(error_df)
    
    # Save predictions
    pred_df = pd.DataFrame({
        "text": X_test,
        "true_label": np.where(y_test == 1, "spam", "ham"),
        "pred_label": np.where(primary_preds == 1, "spam", "ham"),
        "spam_probability": primary_proba
    })
    pred_df.to_csv(config.PREDICTIONS_PATH, index=False, encoding="utf-8")
    
    # Save Visualizations
    save_all_visualizations(
        df=df,
        y_test=y_test,
        y_pred=primary_preds,
        y_proba=primary_proba,
        model_comparison_df=comp_df,
        top_spam=top_spam,
        top_ham=top_ham,
        output_dir=config.CHARTS_DIR
    )
    
    # Generate Markdown Report
    generate_markdown_report(
        metrics=primary_metrics,
        cm_breakdown=cm_breakdown,
        model_comp_df=comp_df,
        diagnostics=diagnostics,
        output_path=config.REPORT_PATH
    )
    
    print("=== PIPELINE EXECUTION COMPLETE ===")

if __name__ == "__main__":
    run_pipeline()
