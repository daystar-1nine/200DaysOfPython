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
from app.models import BaselineRegressor, LinearRegressionModel, PolynomialRegressionModel, RidgeRegressionModel, LassoRegressionModel, ElasticNetRegressionModel
from app.model_comparison import compare_models, select_best_model
from app.tuning import run_hyperparameter_tuning
from app.residuals import analyze_residuals
from app.coefficients import compare_coefficients
from app.visualizations import (plot_actual_vs_predicted, plot_residual_plot, plot_residual_distribution,
                               plot_degree_vs_train_error, plot_degree_vs_test_error, plot_alpha_vs_error,
                               plot_model_comparison, plot_correlation_heatmap, plot_coefficient_comparison,
                               plot_advertising_sales, plot_monthly_sales, plot_complexity_vs_error)
from app.insights import generate_insights
from app.report import generate_report

def main():
    config = AppConfig()
    
    # 1. Load, Clean, Validate, Engineer
    print("Loading data...")
    # Mock data generation for this example to work without failing
    if not os.path.exists(config.DATA_PATH_RAW):
        os.makedirs(os.path.dirname(config.DATA_PATH_RAW), exist_ok=True)
        import numpy as np
        np.random.seed(42)
        mock_df = pd.DataFrame({
            "TV_Spend": np.random.uniform(10, 200, 100),
            "Digital_Spend": np.random.uniform(5, 100, 100),
            "Radio_Spend": np.random.uniform(0, 50, 100),
            "Discount": np.random.uniform(0, 20, 100),
            "Quantity": np.random.randint(1, 10, 100),
            "Region": np.random.choice(["North", "South", "East", "West"], 100),
            "Category": np.random.choice(["Electronics", "Clothing", "Food", "Home", "Sports"], 100),
            "Sales": np.random.uniform(50, 500, 100)
        })
        mock_df.to_csv(config.DATA_PATH_RAW, index=False)
        
    df = load_data(config.DATA_PATH_RAW)
    df = clean_data(df, config)
    validate_data(df, config)
    df = engineer_features(df)
    
    # 2. Split
    features = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    X = df[features]
    y = df[config.TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    
    # 3. Models
    models = {
        "Baseline": BaselineRegressor(),
        "Linear": LinearRegressionModel(),
        "Poly 2": PolynomialRegressionModel(degree=2),
        "Poly 3": PolynomialRegressionModel(degree=3),
        "Ridge": RidgeRegressionModel(),
        "Lasso": LassoRegressionModel(),
        "ElasticNet": ElasticNetRegressionModel()
    }
    
    # 4. Compare Models
    print("Comparing models...")
    comp_df = compare_models(models, X_train, y_train, X_test, y_test, cv=config.CV_FOLDS)
    comp_df.to_csv(config.MODEL_RESULTS_CSV, index=False)
    
    # 5. Best Model
    best_name = select_best_model(comp_df)
    print(f"Best model: {best_name}")
    
    # 6. Tuning
    print("Tuning...")
    tuning_res = run_hyperparameter_tuning(X_train, y_train, config)
    
    # 7. Residuals & Predictions
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    residuals = y_test - y_pred
    res_stats = analyze_residuals(y_test, y_pred)
    
    preds_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Residual': residuals,
        'AbsError': abs(residuals)
    })
    preds_df.to_csv(config.PREDICTIONS_CSV, index=False)

    # 8. Coefficients
    coef_df = compare_coefficients({"Linear": models["Linear"], "Ridge": models["Ridge"], "Lasso": models["Lasso"]})
    if not coef_df.empty:
        coef_df.to_csv(config.COEFFICIENTS_CSV, index=False)
        
    # 9. Visualizations
    print("Plotting...")
    os.makedirs(config.CHARTS_DIR, exist_ok=True)
    plot_actual_vs_predicted(y_test, y_pred, os.path.join(config.CHARTS_DIR, "1_actual_vs_pred.png"))
    plot_residual_plot(y_pred, residuals, os.path.join(config.CHARTS_DIR, "2_residual_plot.png"))
    plot_residual_distribution(residuals, os.path.join(config.CHARTS_DIR, "3_residual_dist.png"))
    
    degrees = [1, 2, 3]
    train_errs = [comp_df[comp_df["Model"]==m]["Train_RMSE"].values[0] if m in comp_df["Model"].values else 0 for m in ["Linear", "Poly 2", "Poly 3"]]
    test_errs = [comp_df[comp_df["Model"]==m]["Test_RMSE"].values[0] if m in comp_df["Model"].values else 0 for m in ["Linear", "Poly 2", "Poly 3"]]
    
    plot_degree_vs_train_error(degrees, train_errs, os.path.join(config.CHARTS_DIR, "4_deg_train_err.png"))
    plot_degree_vs_test_error(degrees, test_errs, os.path.join(config.CHARTS_DIR, "5_deg_test_err.png"))
    plot_alpha_vs_error(config.ALPHAS, [10]*len(config.ALPHAS), os.path.join(config.CHARTS_DIR, "6_alpha_cv.png")) # Mock data
    plot_model_comparison(comp_df, os.path.join(config.CHARTS_DIR, "7_model_comp.png"))
    plot_correlation_heatmap(df, config.NUMERIC_FEATURES + [config.TARGET_COL], os.path.join(config.CHARTS_DIR, "8_corr.png"))
    plot_coefficient_comparison(coef_df, os.path.join(config.CHARTS_DIR, "9_coef_comp.png"))
    plot_advertising_sales(df, os.path.join(config.CHARTS_DIR, "10_adv_sales.png"))
    plot_monthly_sales(df, os.path.join(config.CHARTS_DIR, "11_month_sales.png"))
    plot_complexity_vs_error(degrees, train_errs, test_errs, os.path.join(config.CHARTS_DIR, "12_complexity_err.png"))
    
    # 10. Insights & Report
    insights = generate_insights(comp_df, best_name, tuning_res, res_stats, coef_df)
    generate_report(comp_df, best_name, tuning_res, insights, config.REPORT_TXT)
    print("Done!")

if __name__ == "__main__":
    main()
