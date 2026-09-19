"""Systematic benchmark of the 5 required model/feature configurations."""
import time
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from app.features.tfidf import get_word_tfidf, get_char_tfidf
from app.models.naive_bayes import get_naive_bayes_model
from app.models.logistic_regression import get_logistic_model
from app.models.linear_svm import get_linear_svm_model
from app.evaluation.cross_validation import run_stratified_cv
from app.evaluation.metrics import evaluate_predictions, measure_inference_time

def run_benchmark(X_train, y_train, X_test, y_test) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    from typing import Tuple, Dict, Any
    
    configs = [
        ("Naive Bayes", "Word TF-IDF", get_word_tfidf(ngram_range=(1, 2)), get_naive_bayes_model(alpha=1.0)),
        ("Logistic Regression", "Word TF-IDF", get_word_tfidf(ngram_range=(1, 2)), get_logistic_model(C=1.0)),
        ("Linear SVM", "Word TF-IDF", get_word_tfidf(ngram_range=(1, 2)), get_linear_svm_model(C=1.0, calibrate=True)),
        ("Logistic Regression", "Char TF-IDF", get_char_tfidf(ngram_range=(3, 5)), get_logistic_model(C=1.0)),
        ("Linear SVM", "Char TF-IDF", get_char_tfidf(ngram_range=(3, 5)), get_linear_svm_model(C=1.0, calibrate=True)),
    ]
    
    table_rows = []
    trained_pipes = {}
    
    for model_name, feat_name, vec, clf in configs:
        pipe_key = f"{model_name} ({feat_name})"
        pipe = Pipeline([("tfidf", vec), ("classifier", clf)])
        
        # 1. 5-Fold Stratified Cross Validation on Training Data
        cv_mean, cv_std = run_stratified_cv(pipe, X_train, y_train, n_splits=5)
        
        # 2. Fit on full training data & measure training time
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time_ms = (time.perf_counter() - t0) * 1000
        
        # 3. Predict on held-out test data
        preds = pipe.predict(X_test)
        
        # Get probability / score
        if hasattr(pipe.named_steps["classifier"], "predict_proba"):
            scores = pipe.predict_proba(X_test)[:, 1]
        else:
            scores = pipe.decision_function(X_test)
            
        metrics = evaluate_predictions(y_test, preds, scores)
        inf_time_ms = measure_inference_time(pipe, X_test)
        vocab_size = len(pipe.named_steps["tfidf"].vocabulary_)
        
        table_rows.append({
            "Model": model_name,
            "Features": feat_name,
            "CV Mean": cv_mean,
            "CV Std": cv_std,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "ROC-AUC": metrics.get("roc_auc", 0.0),
            "Train Time (ms)": train_time_ms,
            "Inference Time (ms)": inf_time_ms,
            "Vocab Size": vocab_size
        })
        trained_pipes[pipe_key] = {
            "pipeline": pipe,
            "predictions": preds,
            "scores": scores,
            "metrics": metrics
        }
        
    return pd.DataFrame(table_rows), trained_pipes
