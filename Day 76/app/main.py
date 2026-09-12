import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.model_selection import train_test_split
from app.config import AppConfig
from app.loader import load_data
from app.cleaner import clean_data
from app.validator import validate_data
from app.feature_engineering import engineer_features
from app.model import train_logistic_model
from app.prediction import generate_predictions
from app.threshold import evaluate_thresholds, find_optimal_threshold
from app.metrics import calculate_all_metrics
from app.confusion_matrix import compute_confusion_matrix
from app.roc_analysis import compute_roc_curve, find_optimal_roc_threshold
from app.business_cost import cost_curve_analysis
from app.visualizations import (
    plot_churn_distribution, plot_age_vs_churn, plot_charges_vs_churn,
    plot_tenure_vs_churn, plot_contract_churn, plot_internet_churn,
    plot_support_calls_churn, plot_confusion_matrix_heatmap, plot_roc_curve,
    plot_threshold_metrics, plot_probability_distribution
)
from app.insights import generate_insights
from app.report import generate_report

def main():
    config = AppConfig()
    
    # 1. Load Data
    print("Loading data...")
    raw_df = load_data(config)
    
    # 2. Clean Data
    print("Cleaning data...")
    clean_df = clean_data(raw_df, config)
    
    # 3. Validate Data
    print("Validating data...")
    validate_data(clean_df, config)
    
    # 4. Feature Engineering
    print("Engineering features...")
    features_df = engineer_features(clean_df)
    
    # 5. Split Data
    print("Splitting data...")
    X = features_df.drop(columns=[config.TARGET_COL, 'Customer_ID'], errors='ignore')
    y = features_df[config.TARGET_COL]
    
    numeric_features = [col for col in config.NUMERIC_FEATURES if col in X.columns]
    categorical_features = [col for col in config.CATEGORICAL_FEATURES if col in X.columns]
    
    # Add newly engineered features to numeric list if present
    eng_features = ['Avg_Monthly_Usage', 'Charges_Per_Call', 'Support_Risk_Index']
    for f in eng_features:
        if f in X.columns:
            numeric_features.append(f)
            
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    
    from app.preprocessing import build_preprocessor
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    
    # 6. Train Model
    print("Training model...")
    model = train_logistic_model(X_train, y_train, preprocessor=preprocessor)
    coef_df = model.get_coefficients()
    coef_df.to_csv(config.COEFFICIENTS_CSV, index=False)
    
    # 7. Predict
    print("Predicting...")
    pred_df = generate_predictions(model, X_test, y_test, threshold=config.DEFAULT_THRESHOLD)
    pred_df.to_csv(config.PREDICTIONS_CSV, index=False)
    
    # 8. Threshold Evaluation
    print("Evaluating thresholds...")
    threshold_df = evaluate_thresholds(y_test, pred_df['Probability'])
    threshold_df.to_csv(config.THRESHOLD_ANALYSIS_CSV, index=False)
    optimal_threshold = find_optimal_threshold(threshold_df, metric='f1')
    
    # Re-predict with optimal threshold
    pred_df_opt = generate_predictions(model, X_test, y_test, threshold=optimal_threshold)
    
    # 9. Confusion Matrix
    print("Computing confusion matrix...")
    cm_dict = compute_confusion_matrix(y_test, pred_df_opt['Predicted_Class'])
    pd.DataFrame(cm_dict['matrix']).to_csv(config.CONFUSION_MATRIX_CSV, index=False)
    
    # 10. ROC / AUC
    print("Computing ROC...")
    metrics = calculate_all_metrics(y_test, pred_df_opt['Predicted_Class'], pred_df_opt['Probability'])
    pd.DataFrame([metrics]).to_csv(config.METRICS_CSV, index=False)
    fpr, tpr, roc_thresholds = compute_roc_curve(y_test, pred_df_opt['Probability'])
    
    # 11. Business Cost
    print("Analyzing business costs...")
    cost_df = cost_curve_analysis(y_test, pred_df_opt['Probability'], threshold_df['Threshold'])
    
    # 12. Visualizations
    print("Generating visualizations...")
    plot_churn_distribution(clean_df, os.path.join(config.CHARTS_DIR, "churn_distribution.png"))
    if 'Age' in clean_df.columns:
        plot_age_vs_churn(clean_df, os.path.join(config.CHARTS_DIR, "age_vs_churn.png"))
    if 'Monthly_Charges' in clean_df.columns:
        plot_charges_vs_churn(clean_df, os.path.join(config.CHARTS_DIR, "charges_vs_churn.png"))
    if 'Tenure_Months' in clean_df.columns:
        plot_tenure_vs_churn(clean_df, os.path.join(config.CHARTS_DIR, "tenure_vs_churn.png"))
    if 'Contract_Type' in clean_df.columns:
        plot_contract_churn(clean_df, os.path.join(config.CHARTS_DIR, "contract_churn.png"))
    if 'Internet_Service' in clean_df.columns:
        plot_internet_churn(clean_df, os.path.join(config.CHARTS_DIR, "internet_churn.png"))
    if 'Support_Calls' in clean_df.columns:
        plot_support_calls_churn(clean_df, os.path.join(config.CHARTS_DIR, "support_calls_churn.png"))
        
    plot_confusion_matrix_heatmap(cm_dict['matrix'], os.path.join(config.CHARTS_DIR, "confusion_matrix.png"))
    plot_roc_curve(fpr, tpr, metrics.get('ROC_AUC', 0), os.path.join(config.CHARTS_DIR, "roc_curve.png"))
    plot_threshold_metrics(threshold_df, os.path.join(config.CHARTS_DIR, "threshold_metrics.png"))
    plot_probability_distribution(pred_df_opt['Probability'], y_test, os.path.join(config.CHARTS_DIR, "probability_dist.png"))
    
    # 13. Insights & 14. Report
    print("Generating report...")
    insights = generate_insights(metrics, cm_dict, threshold_df, optimal_threshold, coef_df, cost_df)
    generate_report(metrics, cm_dict, optimal_threshold, insights, config.REPORT_TXT)
    
    print("Pipeline complete!")

if __name__ == "__main__":
    main()
