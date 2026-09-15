import os
import sys

# Ensure Day 79 root directory is in sys.path
DAY79_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY79_DIR not in sys.path:
    sys.path.insert(0, DAY79_DIR)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from app.config import AppConfig
from app.loader import load_raw_data
from app.cleaner import clean_dataset
from app.validator import validate_raw_data, validate_cleaned_data
from app.feature_engineering import engineer_features
from app.preprocessing import build_tree_preprocessor, build_linear_preprocessor, split_data
from app.models.logistic_baseline import create_logistic_pipeline
from app.models.decision_tree import create_decision_tree_pipeline
from app.models.random_forest import create_random_forest_pipeline
from app.tuning.cross_validation import evaluate_stratified_cv
from app.tuning.grid_search import run_rf_grid_search
from app.evaluation.metrics import calculate_classification_metrics
from app.evaluation.confusion_matrix import compute_confusion_matrix_breakdown
from app.evaluation.roc import compute_roc_curve_data
from app.evaluation.precision_recall import compute_pr_curve_data
from app.feature_importance import compute_mdi_importance
from app.permutation_importance import compute_permutation_importance
from app.threshold import sweep_thresholds, find_optimal_threshold
from app.business_cost import compute_business_cost, sweep_cost_thresholds, find_minimum_cost_threshold
from app.risk_scoring import assign_risk_tiers
from app.insights import generate_executive_insights
from app.visualizations import (
    plot_confusion_matrices, plot_roc_curves, plot_pr_curves,
    plot_model_benchmarks, plot_mdi_importance, plot_permutation_importance,
    plot_mdi_vs_permutation, plot_threshold_curves, plot_cost_curve,
    plot_oob_error_vs_trees, plot_tree_variance_reduction, plot_cv_boxplots,
    plot_risk_distribution, plot_cost_savings_waterfall
)
from app.report import generate_markdown_report

def main():
    print("==========================================================")
    print("🌲 Day 79 - Random Forest Churn Prediction Engine Pipeline")
    print("==========================================================\n")
    
    cfg = AppConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.charts_dir, exist_ok=True)
    os.makedirs(cfg.metrics_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.processed_train_path), exist_ok=True)
    
    # 1. Load Data
    print("1. Loading raw dataset...")
    df_raw = load_raw_data(cfg.raw_data_path)
    val_raw, errs_raw = validate_raw_data(df_raw)
    if not val_raw:
        print(f"Raw data validation failed: {errs_raw}")
        sys.exit(1)
    print(f"   Loaded {len(df_raw)} records successfully.")
    
    # 2. Clean Data
    print("2. Cleaning dataset...")
    df_clean = clean_dataset(df_raw)
    val_clean, errs_clean = validate_cleaned_data(df_clean)
    if not val_clean:
        print(f"Cleaned data validation failed: {errs_clean}")
        sys.exit(1)
        
    # 3. Feature Engineering
    print("3. Engineering features...")
    df_feat = engineer_features(df_clean)
    
    # 4. Train-Test Split
    feature_cols = cfg.numeric_features + cfg.categorical_features
    X_train, X_test, y_train, y_test = split_data(
        df_feat, feature_cols, cfg.target_column, test_size=cfg.test_size, random_state=cfg.random_state
    )
    
    # Save processed data
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_csv(cfg.processed_train_path, index=False)
    test_df.to_csv(cfg.processed_test_path, index=False)
    print(f"   Train set: {len(X_train)} rows | Test set: {len(X_test)} rows")
    
    # 5. Build Preprocessors and Models
    print("4. Building model pipelines...")
    tree_preproc = build_tree_preprocessor(cfg.numeric_features, cfg.categorical_features)
    linear_preproc = build_linear_preprocessor(cfg.numeric_features, cfg.categorical_features)
    
    pipe_lr = create_logistic_pipeline(linear_preproc, random_state=cfg.random_state)
    pipe_dt = create_decision_tree_pipeline(tree_preproc, max_depth=6, random_state=cfg.random_state)
    pipe_rf = create_random_forest_pipeline(
        tree_preproc, n_estimators=cfg.rf_n_estimators, max_depth=cfg.rf_max_depth,
        min_samples_split=cfg.rf_min_samples_split, min_samples_leaf=cfg.rf_min_samples_leaf,
        max_features=cfg.rf_max_features, bootstrap=True, oob_score=True, random_state=cfg.random_state
    )
    
    # 6. Fit Models
    print("5. Training models...")
    pipe_lr.fit(X_train, y_train)
    pipe_dt.fit(X_train, y_train)
    pipe_rf.fit(X_train, y_train)
    
    # Extract OOB score
    rf_classifier = pipe_rf.named_steps['classifier']
    oob_score = float(rf_classifier.oob_score_)
    print(f"   Random Forest OOB Accuracy Score: {oob_score:.4f}")
    
    # 7. Hyperparameter Tuning (Grid Search on RF)
    print("6. Tuning Random Forest hyperparameters via 5-Fold Stratified CV...")
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [8, 12],
        'classifier__min_samples_split': [2, 5]
    }
    tuned_rf, best_params, best_f1 = run_rf_grid_search(pipe_rf, X_train, y_train, param_grid, cv=5, scoring='f1')
    print(f"   Best params: {best_params} | Best CV F1: {best_f1:.4f}")
    
    # 8. Model Evaluations on Test Set
    print("7. Evaluating benchmark models on test set...")
    models = {
        'Logistic Regression': pipe_lr,
        'Decision Tree': pipe_dt,
        'Random Forest': pipe_rf,
        'Tuned Random Forest': tuned_rf
    }
    
    benchmark_metrics = {}
    test_preds = {}
    test_probs = {}
    cms = {}
    roc_data = {}
    pr_data = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        test_preds[name] = y_pred
        test_probs[name] = y_prob
        
        from sklearn.metrics import confusion_matrix
        cms[name] = confusion_matrix(y_test, y_pred)
        
        metrics = calculate_classification_metrics(y_test.values, y_pred, y_prob)
        benchmark_metrics[name] = metrics
        
        fpr, tpr, _, r_auc = compute_roc_curve_data(y_test.values, y_prob)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': r_auc}
        
        prec, rec, _, p_auc = compute_pr_curve_data(y_test.values, y_prob)
        pr_data[name] = {'precision': prec, 'recall': rec, 'ap': p_auc}
        
        print(f"   {name:20s} -> F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | Acc: {metrics['accuracy']:.4f}")
        
    # Save benchmark metrics to CSV
    df_metrics = pd.DataFrame(benchmark_metrics).T
    df_metrics.to_csv(os.path.join(cfg.metrics_dir, 'model_benchmark_metrics.csv'))
    
    # 9. 5-Fold Cross-Validation Analysis
    print("8. Running 5-Fold Stratified Cross-Validation on core models...")
    cv_summary = {
        'Logistic Regression': evaluate_stratified_cv(pipe_lr, X_train, y_train, cv=5),
        'Decision Tree': evaluate_stratified_cv(pipe_dt, X_train, y_train, cv=5),
        'Random Forest': evaluate_stratified_cv(pipe_rf, X_train, y_train, cv=5)
    }
    
    # CV boxplot records
    cv_records = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, pipe in [('Logistic Regression', pipe_lr), ('Decision Tree', pipe_dt), ('Random Forest', pipe_rf)]:
        scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='f1')
        for s in scores:
            cv_records.append({'Model': name, 'F1_Score': s})
    df_cv_records = pd.DataFrame(cv_records)
    
    # 10. Feature Importance (MDI & Permutation)
    print("9. Computing MDI and Permutation Importance...")
    df_mdi = compute_mdi_importance(pipe_rf, cfg.numeric_features, cfg.categorical_features)
    df_mdi.to_csv(os.path.join(cfg.metrics_dir, 'feature_importance_mdi.csv'), index=False)
    
    df_perm = compute_permutation_importance(pipe_rf, X_test, y_test, n_repeats=10, scoring='f1')
    df_perm.to_csv(os.path.join(cfg.metrics_dir, 'feature_importance_permutation.csv'), index=False)
    
    # 11. Threshold Optimization
    print("10. Sweeping decision thresholds...")
    rf_probs = test_probs['Random Forest']
    df_thresh = sweep_thresholds(y_test.values, rf_probs)
    opt_f1_t, opt_f1_val = find_optimal_threshold(df_thresh, 'F1_Score')
    df_thresh.to_csv(os.path.join(cfg.metrics_dir, 'threshold_sweep.csv'), index=False)
    print(f"   Optimal F1 Threshold: {opt_f1_t:.2f} (F1 = {opt_f1_val:.4f})")
    
    # 12. Business Cost Optimization
    print("11. Computing business costs and savings...")
    cost_df = sweep_cost_thresholds(y_test.values, rf_probs, cost_fp=cfg.cost_fp, cost_fn=cfg.cost_fn)
    cost_df.to_csv(os.path.join(cfg.metrics_dir, 'cost_sweep.csv'), index=False)
    opt_cost_t, min_cost = find_minimum_cost_threshold(cost_df)
    
    default_cost_res = compute_business_cost(y_test.values, (rf_probs >= 0.50).astype(int), cfg.cost_fp, cfg.cost_fn)
    default_cost = default_cost_res['Total_Cost']
    
    unmanaged_cost = sum(y_test == 1) * cfg.cost_fn
    savings = default_cost - min_cost
    pct_savings = (savings / default_cost) * 100.0 if default_cost > 0 else 0.0
    
    cost_summary = {
        'cost_fp': cfg.cost_fp,
        'cost_fn': cfg.cost_fn,
        'unmanaged_cost': unmanaged_cost,
        'default_cost': default_cost,
        'optimal_threshold': opt_cost_t,
        'optimal_cost': min_cost,
        'savings': savings,
        'pct_savings': pct_savings
    }
    pd.DataFrame([cost_summary]).to_csv(os.path.join(cfg.metrics_dir, 'business_cost_summary.csv'), index=False)
    print(f"   Default Cost (t=0.50): INR {default_cost:,.2f} | Optimal Cost (t={opt_cost_t:.2f}): INR {min_cost:,.2f}")
    print(f"   Savings: INR {savings:,.2f} ({pct_savings:.1f}%)")
    
    # 13. Risk Scoring
    print("12. Scoring customer risk tiers...")
    df_risk = assign_risk_tiers(rf_probs)
    df_risk['Customer_ID'] = df_clean.loc[X_test.index, 'Customer_ID'].values
    df_risk['Actual_Churn'] = y_test.values
    df_risk.to_csv(os.path.join(cfg.metrics_dir, 'customer_risk_scores.csv'), index=False)
    
    # 14. OOB Error Curve Simulation
    print("13. Computing OOB error curve across tree counts...")
    tree_counts = [10, 25, 50, 75, 100, 125, 150]
    oob_errors = []
    for tc in tree_counts:
        temp_rf = create_random_forest_pipeline(tree_preproc, n_estimators=tc, random_state=42)
        temp_rf.fit(X_train, y_train)
        oob_acc = temp_rf.named_steps['classifier'].oob_score_
        oob_errors.append(1.0 - oob_acc)
        
    # Tree variance reduction probability sampling
    transformed_X_test = tree_preproc.transform(X_test)
    indiv_probs = np.array([tree.predict_proba(transformed_X_test)[:, 1] for tree in rf_classifier.estimators_]).T
    
    # 15. Generate Visualizations (14 charts)
    print("14. Generating 14 analytical visualizations...")
    plot_confusion_matrices(cms, os.path.join(cfg.charts_dir, '01_confusion_matrix_comparison.png'))
    plot_roc_curves(roc_data, os.path.join(cfg.charts_dir, '02_roc_curves_comparison.png'))
    base_rate = float(np.mean(y_test))
    plot_pr_curves(pr_data, base_rate, os.path.join(cfg.charts_dir, '03_pr_curves_comparison.png'))
    plot_model_benchmarks(benchmark_metrics, os.path.join(cfg.charts_dir, '04_model_metrics_benchmark.png'))
    plot_mdi_importance(df_mdi, os.path.join(cfg.charts_dir, '05_feature_importance_mdi.png'))
    plot_permutation_importance(df_perm, os.path.join(cfg.charts_dir, '06_permutation_importance.png'))
    plot_mdi_vs_permutation(df_mdi, df_perm, os.path.join(cfg.charts_dir, '07_mdi_vs_permutation_comparison.png'))
    plot_threshold_curves(df_thresh, opt_f1_t, opt_f1_val, os.path.join(cfg.charts_dir, '08_threshold_vs_f1_precision_recall.png'))
    plot_cost_curve(cost_df, opt_cost_t, min_cost, default_cost, os.path.join(cfg.charts_dir, '09_business_cost_curve.png'))
    plot_oob_error_vs_trees(tree_counts, oob_errors, os.path.join(cfg.charts_dir, '10_oob_error_vs_trees.png'))
    plot_tree_variance_reduction(indiv_probs, rf_probs, os.path.join(cfg.charts_dir, '11_tree_variance_reduction.png'))
    plot_cv_boxplots(df_cv_records, os.path.join(cfg.charts_dir, '12_stratified_cv_boxplots.png'))
    plot_risk_distribution(df_risk, os.path.join(cfg.charts_dir, '13_customer_risk_distribution.png'))
    plot_cost_savings_waterfall(unmanaged_cost, default_cost, min_cost, os.path.join(cfg.charts_dir, '14_cost_savings_waterfall.png'))
    print("   All 14 visualizations saved to output/charts/")
    
    # 16. Generate Final Report
    print("15. Generating markdown report...")
    report_path = os.path.join(cfg.output_dir, 'Day79_Final_Report.md')
    generate_markdown_report(
        benchmark_metrics=benchmark_metrics,
        cv_summary=cv_summary,
        oob_score=oob_score,
        best_params=best_params,
        top_mdi_df=df_mdi,
        cost_summary=cost_summary,
        output_file=report_path
    )
    
    # 17. Executive Insights
    top_features = list(df_mdi['Feature'].head(3))
    insights = generate_executive_insights(
        benchmark_metrics=benchmark_metrics,
        rf_oob_score=oob_score,
        top_mdi_features=top_features,
        cost_comparison=cost_summary
    )
    print("\n" + insights)
    print("==========================================================")
    print(" Pipeline execution finished successfully!")
    print("==========================================================")

if __name__ == '__main__':
    main()
