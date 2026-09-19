"""Main orchestration pipeline for Day 102 Benchmark."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import config
from app.data.loader import load_data
from app.data.validator import validate_data
from app.data.audit import audit_dataset
from app.preprocessing.cleaner import clean_punctuation
from app.experiments.feature_experiments import run_feature_experiments
from app.experiments.model_benchmark import run_benchmark
from app.analysis.feature_analysis import extract_top_features
from app.analysis.error_analysis import extract_classification_errors
from app.visualization.charts import generate_all_charts
from app.report import generate_benchmark_report

def main():
    print("=== STARTING DAY 102 NLP CLASSIFICATION BENCHMARK ENGINE ===")
    
    # 1. Load & Validate Data
    raw_df = load_data(config.DATA_RAW)
    validation = validate_data(raw_df)
    if not validation["is_valid"]:
        raise ValueError(f"Data validation failed: {validation['issues']}")
        
    audit_results = audit_dataset(raw_df)
    print(f"Data Audit Complete: {audit_results['total_records']} records, classes: {audit_results['class_distribution']}")
    
    # Preprocess
    df = raw_df.copy()
    df["clean_text"] = df["text"].apply(clean_punctuation)
    df["char_length"] = df["clean_text"].apply(len)
    df["word_count"] = df["clean_text"].apply(lambda s: len(s.split()))
    df.to_csv(config.DATA_PROCESSED, index=False, encoding="utf-8")
    
    # 2. Train/Test Split (Stratified)
    X = df["clean_text"].values
    y = np.where(df["label"] == "spam", 1, 0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    print(f"Split: Train={len(X_train)}, Test={len(X_test)}")
    
    # 3. Feature Experiments
    print("Running Feature Experiments...")
    feat_exp_df = run_feature_experiments(X_train, y_train)
    
    # 4. Model Benchmark (5 Configurations with 5-Fold Stratified CV)
    print("Running Model Benchmark Suite...")
    benchmark_df, trained_pipes = run_benchmark(X_train, y_train, X_test, y_test)
    benchmark_df.to_csv(config.METRICS_PATH, index=False)
    
    # 5. Feature Analysis
    print("Extracting Feature Importances...")
    best_pipe_key = "Logistic Regression (Word TF-IDF)"
    top_spam, top_ham = extract_top_features(trained_pipes[best_pipe_key]["pipeline"])
    feature_analysis_df = pd.concat([top_spam, top_ham], ignore_index=True)
    feature_analysis_df.to_csv(config.FEATURE_ANALYSIS_PATH, index=False)
    
    # 6. Error Analysis
    print("Extracting Classification Errors...")
    all_errors = []
    for pipe_key, data in trained_pipes.items():
        err_df = extract_classification_errors(X_test, y_test, data["predictions"], data["scores"], model_name=pipe_key)
        if not err_df.empty:
            all_errors.append(err_df)
    errors_df = pd.concat(all_errors, ignore_index=True) if all_errors else pd.DataFrame(columns=["text", "actual", "predicted", "model", "confidence_or_score", "error_type"])
    errors_df.to_csv(config.ERRORS_PATH, index=False)
    
    # Save Predictions
    pred_df = pd.DataFrame({
        "text": X_test,
        "actual": np.where(y_test == 1, "spam", "ham"),
        "predicted_lr_word": np.where(trained_pipes["Logistic Regression (Word TF-IDF)"]["predictions"] == 1, "spam", "ham"),
        "score_lr_word": trained_pipes["Logistic Regression (Word TF-IDF)"]["scores"]
    })
    pred_df.to_csv(config.PREDICTIONS_PATH, index=False, encoding="utf-8")
    
    # 7. Visualizations (15 Charts)
    print("Generating 15 Analytical Charts...")
    generate_all_charts(
        df=df,
        benchmark_df=benchmark_df,
        feature_exp_df=feat_exp_df,
        trained_pipes=trained_pipes,
        top_spam=top_spam,
        top_ham=top_ham,
        y_test=y_test,
        errors_df=errors_df,
        output_dir=config.CHARTS_DIR
    )
    
    # 8. Report
    print("Generating Report...")
    generate_benchmark_report(
        audit_data=audit_results,
        benchmark_df=benchmark_df,
        feature_exp_df=feat_exp_df,
        errors_df=errors_df,
        output_path=config.REPORT_PATH
    )
    
    print("=== DAY 102 BENCHMARK ENGINE COMPLETE ===")

if __name__ == "__main__":
    main()
