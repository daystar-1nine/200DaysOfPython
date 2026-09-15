import os
import sys

# Ensure Day 80 root is on sys.path
DAY80_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY80_DIR not in sys.path:
    sys.path.insert(0, DAY80_DIR)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

from app.config import AppConfig
from app.data.loader import load_raw_data
from app.data.cleaner import clean_dataset, generate_data_quality_report
from app.data.validator import validate_raw_data, validate_processed_data
from app.preprocessing.pipeline import build_preprocessor, split_dataset
from app.models.baseline import create_baseline_pipeline
from app.models.logistic import create_logistic_pipeline
from app.models.decision_tree import create_decision_tree_pipeline
from app.models.random_forest import create_random_forest_pipeline
from app.tuning.grids import get_logistic_param_grid, get_decision_tree_param_grid, get_random_forest_param_grid
from app.tuning.search import run_model_grid_search
from app.evaluation.cross_validation import run_stratified_cv
from app.evaluation.metrics import evaluate_classifier_metrics
from app.evaluation.confusion_matrix import compute_confusion_breakdown
from app.evaluation.roc import calculate_roc_curve
from app.evaluation.precision_recall import calculate_pr_curve
from app.evaluation.calibration import calculate_calibration
from app.threshold import sweep_thresholds, find_best_f1_threshold
from app.business_cost import compute_total_cost, sweep_cost_curve, find_optimal_cost_threshold
from app.feature_importance import compute_tree_feature_importance
from app.permutation_importance import compute_permutation_importance
from app.model_comparison import build_model_comparison_table, select_best_model
from app.insights import generate_business_insights
from app.visualizations import (
    plot_churn_distribution, plot_churn_by_category, plot_tenure_distribution,
    plot_charges_by_churn, plot_cv_metric_comparison, plot_test_metrics_grouped,
    plot_train_vs_val, plot_confusion_matrix, plot_roc_curves, plot_pr_curves,
    plot_threshold_curves, plot_cost_curve, plot_feature_importance_bar,
    plot_permutation_bar, plot_calibration_curves
)
from app.report import generate_milestone_markdown_report

def main():
    print("==========================================================")
    print("🚀 Day 80: End-to-End ML Model Selection Engine Milestone")
    print("==========================================================\n")
    
    cfg = AppConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.charts_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.processed_data_path), exist_ok=True)
    
    # 1. Load Data
    print("1. Loading raw customer churn dataset...")
    df_raw = load_raw_data(cfg.raw_data_path)
    val_raw, errs_raw = validate_raw_data(df_raw)
    if not val_raw:
        print(f"Raw data validation failed: {errs_raw}")
        sys.exit(1)
    print(f"   Successfully loaded {len(df_raw)} records.")
    
    # 2. Data Quality & Cleaning
    print("2. Generating data quality report & cleaning dataset...")
    df_quality = generate_data_quality_report(df_raw)
    df_quality.to_csv(os.path.join(cfg.output_dir, 'data_quality_report.csv'), index=False)
    
    df_clean = clean_dataset(df_raw)
    val_proc, errs_proc = validate_processed_data(df_clean)
    if not val_proc:
        print(f"Processed data validation failed: {errs_proc}")
        sys.exit(1)
    df_clean.to_csv(cfg.processed_data_path, index=False)
    print(f"   Cleaned dataset saved: {len(df_clean)} records.")
    
    # 3. Train-Test Split (Leakage Prevention)
    print("3. Performing stratified train/test split (80% Train, 20% Final Test)...")
    feature_cols = cfg.numeric_features + cfg.categorical_features
    X_train, X_test, y_train, y_test = split_dataset(
        df_clean, feature_cols, cfg.target_column, test_size=cfg.test_size, random_state=cfg.random_state
    )
    print(f"   Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    # 4. Build Preprocessors
    print("4. Constructing isolated ColumnTransformer preprocessors...")
    linear_preproc = build_preprocessor(cfg.numeric_features, cfg.categorical_features, scale_numeric=True)
    tree_preproc = build_preprocessor(cfg.numeric_features, cfg.categorical_features, scale_numeric=False)
    
    # 5. Build Pipelines
    print("5. Initializing candidate model architectures...")
    pipe_dummy = create_baseline_pipeline(linear_preproc)
    pipe_log = create_logistic_pipeline(linear_preproc, random_state=cfg.random_state)
    pipe_tree = create_decision_tree_pipeline(tree_preproc, random_state=cfg.random_state)
    pipe_forest = create_random_forest_pipeline(tree_preproc, random_state=cfg.random_state)
    
    raw_candidates = {
        'Baseline (Dummy)': pipe_dummy,
        'Logistic Regression': pipe_log,
        'Decision Tree': pipe_tree,
        'Random Forest': pipe_forest
    }
    
    # 6. Stratified 5-Fold Cross Validation on Raw Pipelines
    print("6. Conducting 5-Fold Stratified Cross-Validation on all candidates...")
    cv_results = {}
    for name, pipe in raw_candidates.items():
        print(f"   Evaluating CV for: {name}...")
        cv_results[name] = run_stratified_cv(pipe, X_train, y_train, cv=5, random_state=cfg.random_state)
        roc_m = cv_results[name]['roc_auc']['test_mean']
        roc_s = cv_results[name]['roc_auc']['test_std']
        print(f"     -> CV ROC-AUC: {roc_m:.4f} (±{roc_s:.4f})")
        
    # Save CV results to CSV
    cv_records = []
    for model_name, metrics in cv_results.items():
        for m_name, vals in metrics.items():
            cv_records.append({
                'Model': model_name,
                'Metric': m_name,
                'Train_Mean': vals['train_mean'],
                'Test_Mean': vals['test_mean'],
                'Test_Std': vals['test_std']
            })
    pd.DataFrame(cv_records).to_csv(os.path.join(cfg.output_dir, 'cross_validation_results.csv'), index=False)
    
    # 7. Hyperparameter Tuning
    print("7. Tuning hyperparameters via GridSearchCV on training set...")
    pipe_log_tuned, best_log_params, _ = run_model_grid_search(pipe_log, X_train, y_train, get_logistic_param_grid(), cv=5)
    pipe_tree_tuned, best_tree_params, _ = run_model_grid_search(pipe_tree, X_train, y_train, get_decision_tree_param_grid(), cv=5)
    pipe_forest_tuned, best_forest_params, _ = run_model_grid_search(pipe_forest, X_train, y_train, get_random_forest_param_grid(), cv=5)
    
    # Fit dummy on train
    pipe_dummy.fit(X_train, y_train)
    
    models_to_test = {
        'Baseline (Dummy)': pipe_dummy,
        'Logistic Regression': pipe_log_tuned,
        'Decision Tree': pipe_tree_tuned,
        'Random Forest': pipe_forest_tuned
    }
    
    # 8. Evaluate on Untouched Final Test Set
    print("8. Evaluating tuned candidates ONCE on untouched final test set...")
    test_results = {}
    test_predictions = pd.DataFrame({'Actual_Churn': y_test.values})
    cms = {}
    roc_curves = {}
    pr_curves = {}
    calib_curves = {}
    business_costs = {}
    
    for name, model in models_to_test.items():
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = np.zeros_like(y_pred, dtype=float)
            
        test_predictions[f'{name}_Pred'] = y_pred
        test_predictions[f'{name}_Prob'] = y_prob
        
        # Metrics
        m = evaluate_classifier_metrics(y_test.values, y_pred, y_prob)
        test_results[name] = m
        
        # Confusion breakdown
        cms[name] = compute_confusion_breakdown(y_test.values, y_pred)
        
        # ROC & PR curves
        fpr, tpr, _, r_auc = calculate_roc_curve(y_test.values, y_prob)
        roc_curves[name] = {'fpr': fpr, 'tpr': tpr, 'auc': r_auc}
        
        prec, rec, _, p_ap = calculate_pr_curve(y_test.values, y_prob)
        pr_curves[name] = {'precision': prec, 'recall': rec, 'ap': p_ap}
        
        # Calibration (skip dummy)
        if name != 'Baseline (Dummy)':
            p_true, p_pred, brier = calculate_calibration(y_test.values, y_prob)
            calib_curves[name] = {'true': p_true, 'pred': p_pred, 'brier': brier}
            
        # Business cost at default 0.50
        cost_res = compute_total_cost(y_test.values, y_pred, cfg.cost_fp, cfg.cost_fn)
        business_costs[name] = cost_res['Total_Cost']
        
        print(f"   {name:22s} -> Test ROC-AUC: {m.get('roc_auc', 0.0):.4f} | F1: {m['f1']:.4f} | Cost: INR {business_costs[name]:,.0f}")
        
    test_predictions.to_csv(os.path.join(cfg.output_dir, 'predictions.csv'), index=False)
    pd.DataFrame(test_results).T.to_csv(os.path.join(cfg.output_dir, 'test_results.csv'))
    
    # 9. Model Comparison Matrix & Champion Selection
    print("9. Assembling standardized model comparison matrix...")
    df_comparison = build_model_comparison_table(cv_results, test_results, business_costs)
    df_comparison.to_csv(os.path.join(cfg.output_dir, 'model_comparison.csv'), index=False)
    champion_model = select_best_model(df_comparison, 'Test_ROC_AUC')
    print(f"   ★ CHAMPION MODEL SELECTED: {champion_model}")
    
    # 10. Threshold & Business Cost Optimization for Champion Model
    print("10. Sweeping decision thresholds & optimizing financial business cost...")
    champ_pipe = models_to_test[champion_model]
    champ_prob = test_predictions[f'{champion_model}_Prob'].values
    
    df_thresh = sweep_thresholds(y_test.values, champ_prob)
    df_thresh.to_csv(os.path.join(cfg.output_dir, 'threshold_analysis.csv'), index=False)
    best_f1_t, best_f1 = find_best_f1_threshold(df_thresh)
    
    cost_df = sweep_cost_curve(y_test.values, champ_prob, cost_fp=cfg.cost_fp, cost_fn=cfg.cost_fn)
    cost_df.to_csv(os.path.join(cfg.output_dir, 'business_cost_analysis.csv'), index=False)
    opt_cost_t, min_cost = find_optimal_cost_threshold(cost_df)
    
    default_cost = business_costs[champion_model]
    savings = default_cost - min_cost
    pct_savings = (savings / default_cost) * 100.0 if default_cost > 0 else 0.0
    
    cost_summary = {
        'default_cost': default_cost,
        'optimal_threshold': opt_cost_t,
        'optimal_cost': min_cost,
        'savings': savings,
        'pct_savings': pct_savings,
        'best_f1_threshold': best_f1_t,
        'best_f1_score': best_f1
    }
    print(f"   Default Threshold (0.50) Cost: INR {default_cost:,.2f}")
    print(f"   Optimal Threshold ({opt_cost_t:.2f}) Cost: INR {min_cost:,.2f} (Savings: INR {savings:,.2f} / {pct_savings:.1f}%)")
    
    # 11. Feature Importance & Explainability
    print("11. Computing MDI & Permutation Feature Importance for Champion...")
    df_mdi = compute_tree_feature_importance(pipe_forest_tuned, cfg.numeric_features, cfg.categorical_features)
    df_mdi.to_csv(os.path.join(cfg.output_dir, 'feature_importance.csv'), index=False)
    
    df_perm = compute_permutation_importance(pipe_forest_tuned, X_test, y_test, scoring='roc_auc')
    df_perm.to_csv(os.path.join(cfg.output_dir, 'permutation_importance.csv'), index=False)
    
    # 12. Generate 17 Visualizations
    print("12. Generating 17 publication-grade analytical visualizations...")
    plot_churn_distribution(df_clean, os.path.join(cfg.charts_dir, '01_churn_distribution.png'))
    plot_churn_by_category(df_clean, 'Contract_Type', 'Churn Rate by Contract Type', os.path.join(cfg.charts_dir, '02_churn_by_contract.png'))
    plot_churn_by_category(df_clean, 'Internet_Service', 'Churn Rate by Internet Service', os.path.join(cfg.charts_dir, '03_churn_by_internet_service.png'))
    plot_tenure_distribution(df_clean, os.path.join(cfg.charts_dir, '04_tenure_distribution.png'))
    plot_charges_by_churn(df_clean, os.path.join(cfg.charts_dir, '05_monthly_charges_by_churn.png'))
    plot_cv_metric_comparison(cv_results, 'roc_auc', '5-Fold CV Mean ROC-AUC Comparison', os.path.join(cfg.charts_dir, '06_cv_roc_auc_comparison.png'))
    plot_cv_metric_comparison(cv_results, 'average_precision', '5-Fold CV Mean Average Precision Comparison', os.path.join(cfg.charts_dir, '07_cv_average_precision_comparison.png'))
    plot_test_metrics_grouped(df_comparison, os.path.join(cfg.charts_dir, '08_test_metric_comparison.png'))
    plot_train_vs_val(cv_results, os.path.join(cfg.charts_dir, '09_train_vs_val_performance.png'))
    plot_confusion_matrix(cms[champion_model], champion_model, os.path.join(cfg.charts_dir, '10_confusion_matrix_best_model.png'))
    plot_roc_curves(roc_curves, os.path.join(cfg.charts_dir, '11_roc_curves_all_models.png'))
    plot_pr_curves(pr_curves, float(np.mean(y_test)), os.path.join(cfg.charts_dir, '12_precision_recall_curves.png'))
    plot_threshold_curves(df_thresh, best_f1_t, os.path.join(cfg.charts_dir, '13_threshold_vs_precision_recall.png'))
    plot_cost_curve(cost_df, opt_cost_t, min_cost, default_cost, os.path.join(cfg.charts_dir, '14_threshold_vs_business_cost.png'))
    plot_feature_importance_bar(df_mdi, 'Random Forest MDI Gini Importance', os.path.join(cfg.charts_dir, '15_feature_importance_mdi.png'))
    plot_permutation_bar(df_perm, os.path.join(cfg.charts_dir, '16_permutation_importance.png'))
    plot_calibration_curves(calib_curves, os.path.join(cfg.charts_dir, '17_probability_calibration_curve.png'))
    print("   All 17 visualization charts generated.")
    
    # 13. Milestone Report & Executive Insights
    print("13. Generating executive summary and milestone report...")
    report_file = os.path.join(cfg.output_dir, 'Day80_Final_Report.md')
    generate_milestone_markdown_report(df_comparison, champion_model, df_mdi, cost_summary, report_file)
    
    top_features = list(df_mdi['Feature'].head(3))
    insights = generate_business_insights(df_comparison, champion_model, top_features, cost_summary)
    print("\n" + insights)
    print("==========================================================")
    print(" End-to-End ML Model Selection Pipeline Completed!")
    print("==========================================================")

if __name__ == '__main__':
    main()
