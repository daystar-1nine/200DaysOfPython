"""
Day 73 - Sales Prediction Regression Engine CLI Entry Point
Orchestrates data loading, cleaning, validation, EDA, training, evaluation, charts, and report generation.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from app.config import AppConfig
    from app.loader import load_csv
    from app.cleaner import clean_sales_data
    from app.validator import validate_data
    from app.eda import compute_eda
    from app.regression import SimpleLinearRegressor
    from app.predictions import generate_scenario_predictions
    from app.metrics import evaluate_predictions
    from app.residuals import analyze_residuals
    from app.visualizations import generate_all_charts
    from app.insights import generate_business_insights
    from app.report import format_ascii_report, export_artifacts
except ImportError:
    from config import AppConfig
    from loader import load_csv
    from cleaner import clean_sales_data
    from validator import validate_data
    from eda import compute_eda
    from regression import SimpleLinearRegressor
    from predictions import generate_scenario_predictions
    from metrics import evaluate_predictions
    from residuals import analyze_residuals
    from visualizations import generate_all_charts
    from insights import generate_business_insights
    from report import format_ascii_report, export_artifacts

def run_pipeline(csv_path: str = None) -> int:
    config = AppConfig()
    if csv_path:
        config.DATA_PATH_RAW = csv_path
        
    print("=" * 70)
    print("         SALES PREDICTION REGRESSION ENGINE PIPELINE         ")
    print("=" * 70)
    print(f"1. Loading dataset from: {config.DATA_PATH_RAW}")
    
    # 1. Load, Clean, and Validate
    raw_df = load_csv(config.DATA_PATH_RAW)
    cleaned_df = clean_sales_data(raw_df, config.FEATURE_COL, config.TARGET_COL)
    val_info = validate_data(cleaned_df, config.FEATURE_COL, config.TARGET_COL)
    print(f"   Cleaned records: {len(cleaned_df)} (Validation: PASSED)")
    
    # Save cleaned dataset to processed directory
    cleaned_df.to_csv(config.DATA_PATH_PROCESSED, index=False)
    
    # 2. EDA
    print("2. Performing Exploratory Data Analysis...")
    eda_dict = compute_eda(cleaned_df, config.FEATURE_COL, config.TARGET_COL)
    print(f"   Pearson Correlation r: {eda_dict['pearson_r']:.4f} (p = {eda_dict['p_value']:.4e})")
    
    # 3. Train / Test Split
    print("3. Executing 80/20 Train/Test Partition...")
    X = cleaned_df[config.FEATURE_COL].values
    y = cleaned_df[config.TARGET_COL].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    # 4. Model Training
    print("4. Fitting Simple Linear Regression Model...")
    regressor = SimpleLinearRegressor(fit_intercept=True)
    regressor.fit(X_train, y_train)
    equation = regressor.get_equation(config.FEATURE_COL, config.TARGET_COL)
    print(f"   Learned: {equation}")
    
    # 5. Evaluation
    print("5. Evaluating Train and Test Performance...")
    y_pred_train = regressor.predict(X_train)
    y_pred_test = regressor.predict(X_test)
    
    train_metrics = evaluate_predictions(y_train, y_pred_train)
    test_metrics = evaluate_predictions(y_test, y_pred_test)
    
    # 6. Residual Analysis
    print("6. Conducting Residual Diagnostics...")
    residual_info = analyze_residuals(y_test, y_pred_test)
    
    # 7. Scenario Predictions
    print("7. Generating Strategic Scenario Forecasts...")
    predictions_df = generate_scenario_predictions(
        model=regressor,
        budgets=config.SCENARIO_BUDGETS,
        min_x=val_info["min_feature"],
        max_x=val_info["max_feature"]
    )
    pred_50k = float(regressor.predict([50000.0])[0])
    
    # 8. Business Insights
    insights = generate_business_insights(
        eda_dict=eda_dict,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        model_params=regressor.get_params(),
        residual_info=residual_info,
        pred_50k=pred_50k
    )
    
    # 9. Render Visualizations
    print("8. Generating 8 Publication-Grade Visualizations...")
    generate_all_charts(
        df=cleaned_df,
        feature_col=config.FEATURE_COL,
        target_col=config.TARGET_COL,
        model=regressor,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        scenarios_df=predictions_df,
        output_dir=config.OUTPUT_DIR
    )
    
    # 10. Export Reports and Metrics
    metrics_summary_df = pd.DataFrame([
        {"Split": "Train (80%)", "MAE": train_metrics["mae"], "MSE": train_metrics["mse"], "RMSE": train_metrics["rmse"], "R2": train_metrics["r2"]},
        {"Split": "Test (20%)", "MAE": test_metrics["mae"], "MSE": test_metrics["mse"], "RMSE": test_metrics["rmse"], "R2": test_metrics["r2"]}
    ])
    
    report_text = format_ascii_report(
        eda_dict=eda_dict,
        model_params=regressor.get_params(),
        equation=equation,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        residual_info=residual_info,
        predictions_df=predictions_df,
        insights=insights
    )
    
    export_artifacts(config.OUTPUT_DIR, metrics_summary_df, predictions_df, report_text)
    print(f"9. Exported artifacts to: {config.OUTPUT_DIR}")
    print("\n" + report_text)
    print("\nPipeline completed successfully!")
    return 0

if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(run_pipeline(path_arg))
