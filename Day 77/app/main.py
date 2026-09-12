import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from app.config import AppConfig
from app.loader import load_data
from app.cleaner import clean_data
from app.validator import validate_data
from app.feature_engineering import engineer_features
from app.preprocessing import build_preprocessor, get_feature_names
from app.validation.stratified_split import stratified_train_test_split
from app.validation.cross_validation import run_stratified_cv
from app.models.logistic import build_logistic_model
from app.models.balanced_logistic import build_balanced_logistic_model
from app.models.tree_comparison import build_random_forest_model
from app.evaluation.metrics import calculate_binary_metrics, calculate_multiclass_metrics, compare_models
from app.evaluation.confusion_matrix import compute_confusion_matrix
from app.evaluation.roc import compute_binary_roc, compute_multiclass_roc
from app.evaluation.precision_recall import compute_binary_pr, compute_multiclass_pr
from app.evaluation.calibration import compute_calibration
from app.threshold import evaluate_thresholds, find_optimal_threshold_f1, find_optimal_threshold_recall_constrained
from app.business_cost import evaluate_cost_curve, find_optimal_cost_threshold
from app.risk_scoring import score_customers
from app.insights import generate_business_insights
from app.visualizations import (
    plot_churn_distribution, plot_risk_distribution, plot_age_vs_churn,
    plot_charges_vs_churn, plot_tenure_vs_churn, plot_contract_churn,
    plot_binary_confusion_matrix, plot_multiclass_confusion_matrix,
    plot_roc_curve, plot_precision_recall_curve, plot_threshold_metrics,
    plot_threshold_f1, plot_probability_distribution, plot_calibration_curve,
    plot_feature_importance, plot_class_performance
)
from app.report import generate_text_report

def run_pipeline():
    config = AppConfig()
    print("=== Step 1: Loading Raw Data ===")
    df_raw = load_data(config.DATA_PATH_RAW, config)
    print(f"Loaded {len(df_raw)} rows.")
    
    print("=== Step 2: Cleaning Data ===")
    df_clean = clean_data(df_raw, config)
    print(f"Cleaned {len(df_clean)} rows.")
    
    print("=== Step 3: Validating Data ===")
    validate_data(df_clean, config)
    print("Data validated successfully.")
    
    print("=== Step 4: Engineering Features ===")
    df_feat = engineer_features(df_clean)
    print("Features engineered.")
    
    print("=== Step 5: Stratified Data Splits ===")
    # Binary Split
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = stratified_train_test_split(
        df_feat, target_col=config.TARGET_BINARY, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    # Multiclass Split
    X_train_mc, X_test_mc, y_train_mc, y_test_mc = stratified_train_test_split(
        df_feat, target_col=config.TARGET_MULTICLASS, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    print("=== Step 6: Building Pipelines and Training Models ===")
    preprocessor = build_preprocessor(config=config)
    
    # Binary models
    bin_models = {
        "Logistic": build_logistic_model(preprocessor, multi_class="ovr", random_state=config.RANDOM_STATE),
        "Balanced_Logistic": build_balanced_logistic_model(preprocessor, multi_class="ovr", random_state=config.RANDOM_STATE),
        "Random_Forest": build_random_forest_model(preprocessor, random_state=config.RANDOM_STATE)
    }
    for m in bin_models.values():
        m.fit(X_train_bin, y_train_bin)
        
    # Multi-class models
    mc_models = {
        "Logistic_OvR": build_logistic_model(preprocessor, multi_class="ovr", random_state=config.RANDOM_STATE),
        "Balanced_Logistic_OvR": build_balanced_logistic_model(preprocessor, multi_class="ovr", random_state=config.RANDOM_STATE),
        "Random_Forest": build_random_forest_model(preprocessor, random_state=config.RANDOM_STATE)
    }
    for m in mc_models.values():
        m.fit(X_train_mc, y_train_mc)
        
    print("=== Step 7: Stratified K-Fold Cross-Validation ===")
    cv_res_bin = run_stratified_cv(bin_models["Balanced_Logistic"], X_train_bin, y_train_bin, n_splits=config.N_SPLITS_CV)
    print(f"Balanced Logistic Binary CV F1-Macro: {cv_res_bin['mean_f1_macro']:.4f} (+/- {cv_res_bin['std_f1_macro']:.4f})")
    
    print("=== Step 8: Evaluating Models ===")
    bin_comp = compare_models(bin_models, X_test_bin, y_test_bin, is_multiclass=False)
    bin_comp.to_csv(config.BINARY_METRICS_CSV, index=False)
    
    mc_comp = compare_models(mc_models, X_test_mc, y_test_mc, is_multiclass=True)
    mc_comp.to_csv(config.MULTICLASS_METRICS_CSV, index=False)
    
    # Best binary model evaluation
    best_bin = bin_models["Balanced_Logistic"]
    y_prob_bin = best_bin.predict_proba(X_test_bin)[:, 1]
    y_pred_bin = best_bin.predict(X_test_bin)
    
    # Best multiclass model evaluation
    best_mc = mc_models["Balanced_Logistic_OvR"]
    y_prob_mc = best_mc.predict_proba(X_test_mc)
    y_pred_mc = best_mc.predict(X_test_mc)
    
    # Confusion matrices
    cm_bin = compute_confusion_matrix(y_test_bin, y_pred_bin, labels=[0, 1])
    cm_mc = compute_confusion_matrix(y_test_mc, y_pred_mc, labels=config.VALID_RISK_LEVELS)
    
    # ROC and PR curves
    roc_bin = compute_binary_roc(y_test_bin, y_prob_bin)
    roc_mc = compute_multiclass_roc(y_test_mc, y_prob_mc, classes=config.VALID_RISK_LEVELS)
    
    pr_bin = compute_binary_pr(y_test_bin, y_prob_bin)
    pr_mc = compute_multiclass_pr(y_test_mc, y_prob_mc, classes=config.VALID_RISK_LEVELS)
    
    # Calibration
    calib = compute_calibration(y_test_bin, y_prob_bin)
    
    print("=== Step 9: Threshold Tuning & Business Cost Analysis ===")
    threshold_df = evaluate_thresholds(y_test_bin, y_prob_bin)
    threshold_df.to_csv(config.THRESHOLD_ANALYSIS_CSV, index=False)
    
    optimal_t_f1 = find_optimal_threshold_f1(threshold_df)
    optimal_t_recall = find_optimal_threshold_recall_constrained(threshold_df, min_recall=0.80)
    
    cost_df = evaluate_cost_curve(y_test_bin, y_prob_bin, fp_cost=config.FP_COST, fn_cost=config.FN_COST)
    cost_df.to_csv(config.BUSINESS_COST_CSV, index=False)
    optimal_t_cost = find_optimal_cost_threshold(cost_df)
    
    print(f"Optimal Thresholds -> F1: {optimal_t_f1:.2f}, Recall>=80%: {optimal_t_recall:.2f}, Min Cost: {optimal_t_cost:.2f}")
    
    print("=== Step 10: Customer Risk Scoring & Prescriptions ===")
    test_cids = X_test_bin["Customer_ID"].values
    risk_scores = score_customers(test_cids, y_prob_bin, y_pred_mc)
    risk_scores.to_csv(config.RISK_SCORES_CSV, index=False)
    
    # Predictions export
    preds_df = pd.DataFrame({
        "Customer_ID": test_cids,
        "True_Churn": y_test_bin.values,
        "Predicted_Churn_Prob": y_prob_bin.round(4),
        "Predicted_Churn_Class": y_pred_bin,
        "True_Risk_Level": y_test_mc.values,
        "Predicted_Risk_Level": y_pred_mc
    })
    preds_df.to_csv(config.PREDICTIONS_CSV, index=False)
    
    print("=== Step 11: Generating Visualizations ===")
    plot_churn_distribution(df_clean, config)
    plot_risk_distribution(df_clean, config)
    plot_age_vs_churn(df_clean, config)
    plot_charges_vs_churn(df_clean, config)
    plot_tenure_vs_churn(df_clean, config)
    plot_contract_churn(df_clean, config)
    plot_binary_confusion_matrix(cm_bin, config)
    plot_multiclass_confusion_matrix(cm_mc, config)
    plot_roc_curve(roc_bin, roc_mc, config)
    plot_precision_recall_curve(pr_bin, pr_mc, config)
    plot_threshold_metrics(threshold_df, config)
    plot_threshold_f1(threshold_df, cost_df, config)
    plot_probability_distribution(y_test_bin, y_prob_bin, config)
    plot_calibration_curve(calib, config)
    
    # Feature importance plot
    feat_names = get_feature_names(best_bin.named_steps["preprocessor"], config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES)
    plot_feature_importance(best_bin, feat_names, config)
    
    # Class performance plot
    mc_metrics = calculate_multiclass_metrics(y_test_mc, y_pred_mc, y_prob_mc, labels=config.VALID_RISK_LEVELS)
    plot_class_performance(mc_metrics, config)
    
    print("=== Step 12: Generating Executive Report & Business Insights ===")
    insights = generate_business_insights(bin_comp, cost_df, df_clean, optimal_t_cost, optimal_t_f1)
    report_text = generate_text_report(
        bin_comp, mc_comp, threshold_df, cost_df,
        optimal_t_f1, optimal_t_cost, insights, config
    )
    print("Pipeline completed successfully!")
    print(report_text)

if __name__ == "__main__":
    run_pipeline()
