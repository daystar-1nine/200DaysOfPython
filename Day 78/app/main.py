import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from sklearn.model_selection import train_test_split

from app.config import AppConfig
from app.loader import load_data
from app.cleaner import clean_data
from app.validator import validate_data
from app.feature_engineering import engineer_features
from app.preprocessing import build_tree_preprocessor, build_scaled_preprocessor, get_transformed_feature_names
from app.models.decision_tree import build_decision_tree_model
from app.models.logistic_baseline import build_logistic_baseline_model
from app.tuning.cross_validation import run_stratified_cv, evaluate_depth_curve
from app.tuning.grid_search import tune_decision_tree
from app.evaluation.metrics import calculate_metrics, compare_models
from app.evaluation.confusion_matrix import compute_confusion_matrix
from app.evaluation.roc import compute_roc
from app.evaluation.precision_recall import compute_pr
from app.feature_importance import extract_feature_importance
from app.permutation_importance import compute_permutation_importance
from app.decision_rules import export_tree_rules, explain_customer_path
from app.threshold import evaluate_thresholds, find_optimal_threshold_f1
from app.business_cost import evaluate_cost_curve, find_optimal_cost_threshold
from app.risk_scoring import score_customers
from app.insights import generate_business_insights
from app.visualizations import (
    plot_churn_distribution, plot_contract_churn, plot_internet_churn,
    plot_charges_churn, plot_tenure_churn, plot_decision_tree_graph,
    plot_feature_importance_bar, plot_permutation_importance_bar,
    plot_confusion_matrix_heatmap, plot_roc_curve_chart,
    plot_pr_curve_chart, plot_depth_vs_score_chart
)
from app.report import generate_executive_report

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
    
    print("=== Step 5: Stratified Train/Test Split ===")
    y = df_feat[config.TARGET_COL]
    X = df_feat.drop(columns=[config.TARGET_COL])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    print("=== Step 6: Building Pipelines ===")
    tree_prep = build_tree_preprocessor(config=config)
    scaled_prep = build_scaled_preprocessor(config=config)
    
    # 1. Unconstrained Gini Tree
    tree_gini = build_decision_tree_model(tree_prep, criterion="gini", random_state=config.RANDOM_STATE)
    # 2. Unconstrained Entropy Tree
    tree_entropy = build_decision_tree_model(tree_prep, criterion="entropy", random_state=config.RANDOM_STATE)
    # 3. Logistic Regression Baseline
    logistic_baseline = build_logistic_baseline_model(scaled_prep, random_state=config.RANDOM_STATE)
    
    tree_gini.fit(X_train, y_train)
    tree_entropy.fit(X_train, y_train)
    logistic_baseline.fit(X_train, y_train)
    
    print("=== Step 7: Tree Depth Curve Analysis ===")
    depth_curve_df = evaluate_depth_curve(tree_prep, X_train, y_train, X_test, y_test, random_state=config.RANDOM_STATE)
    plot_depth_vs_score_chart(depth_curve_df, config)
    
    print("=== Step 8: Hyperparameter Tuning via Stratified Grid Search ===")
    base_tune_pipeline = build_decision_tree_model(tree_prep, random_state=config.RANDOM_STATE)
    param_grid = {
        "classifier__criterion": ["gini", "entropy"],
        "classifier__max_depth": [3, 4, 5, 6, 8],
        "classifier__min_samples_split": [2, 5, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 5, 10]
    }
    best_tree, best_params, best_cv_score, cv_results_df = tune_decision_tree(
        base_tune_pipeline, X_train, y_train, param_grid=param_grid,
        n_splits=config.N_SPLITS_CV, scoring="f1", random_state=config.RANDOM_STATE
    )
    cv_results_df.to_csv(config.HYPERPARAM_RESULTS_CSV, index=False)
    print(f"Best CV F1: {best_cv_score:.4f} with params: {best_params}")
    
    print("=== Step 9: Model Comparison on Test Set ===")
    models = {
        "Logistic_Regression": logistic_baseline,
        "Decision_Tree_Gini_Unconstrained": tree_gini,
        "Decision_Tree_Entropy_Unconstrained": tree_entropy,
        "Tuned_Decision_Tree": best_tree
    }
    model_comp_df = compare_models(models, X_test, y_test)
    model_comp_df.to_csv(config.MODEL_METRICS_CSV, index=False)
    
    print("=== Step 10: Best Model Detailed Evaluation & Explainability ===")
    y_pred = best_tree.predict(X_test)
    y_prob = best_tree.predict_proba(X_test)[:, 1]
    
    cm_data = compute_confusion_matrix(y_test, y_pred)
    roc_data = compute_roc(y_test, y_prob)
    pr_data = compute_pr(y_test, y_prob)
    
    transformed_feature_names = get_transformed_feature_names(
        best_tree.named_steps["preprocessor"], config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES
    )
    
    # Feature Importance (MDI)
    feat_imp_df = extract_feature_importance(best_tree, transformed_feature_names)
    feat_imp_df.to_csv(config.FEATURE_IMPORTANCE_CSV, index=False)
    
    # Permutation Importance
    perm_imp_df = compute_permutation_importance(best_tree, X_test, y_test, transformed_feature_names, scoring="f1", random_state=config.RANDOM_STATE)
    perm_imp_df.to_csv(config.PERMUTATION_IMPORTANCE_CSV, index=False)
    
    # Decision Rules
    rules_text = export_tree_rules(best_tree, transformed_feature_names, max_depth=4)
    with open(config.DECISION_RULES_TXT, "w", encoding="utf-8") as f:
        f.write(rules_text)
        
    # Explain a representative high-risk customer
    sample_row = X_test.iloc[[0]]
    customer_explanation = explain_customer_path(best_tree, sample_row, transformed_feature_names)
    
    print("=== Step 11: Threshold & Financial Cost Analysis ===")
    threshold_df = evaluate_thresholds(y_test, y_prob)
    threshold_df.to_csv(config.THRESHOLD_ANALYSIS_CSV, index=False)
    optimal_t_f1 = find_optimal_threshold_f1(threshold_df)
    
    cost_df = evaluate_cost_curve(y_test, y_prob, fp_cost=config.FP_COST, fn_cost=config.FN_COST)
    cost_df.to_csv(config.BUSINESS_COST_CSV, index=False)
    optimal_t_cost = find_optimal_cost_threshold(cost_df)
    
    # Customer risk scoring
    risk_df = score_customers(X_test["Customer_ID"].values, y_prob, y_pred)
    risk_df.to_csv(config.PREDICTIONS_CSV, index=False)
    
    print("=== Step 12: Visualizations Generation ===")
    plot_churn_distribution(df_clean, config)
    plot_contract_churn(df_clean, config)
    plot_internet_churn(df_clean, config)
    plot_charges_churn(df_clean, config)
    plot_tenure_churn(df_clean, config)
    plot_decision_tree_graph(best_tree, transformed_feature_names, config, max_depth=3)
    plot_feature_importance_bar(feat_imp_df, config)
    plot_permutation_importance_bar(perm_imp_df, config)
    plot_confusion_matrix_heatmap(cm_data, config)
    plot_roc_curve_chart(roc_data, config)
    plot_pr_curve_chart(pr_data, config)
    
    print("=== Step 13: Executive Report Generation ===")
    insights = generate_business_insights(model_comp_df, cost_df, feat_imp_df, df_clean, optimal_t_cost, best_params)
    report = generate_executive_report(
        model_comp_df, best_params, best_cv_score, threshold_df, cost_df,
        optimal_t_f1, optimal_t_cost, insights, customer_explanation, config
    )
    print(report)
    print("Pipeline complete!")

if __name__ == "__main__":
    run_pipeline()
