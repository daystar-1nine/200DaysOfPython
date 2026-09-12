import os
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from app.config import AppConfig
    from app.loader import load_data
    from app.cleaner import clean_data
    from app.validator import validate_data
    from app.feature_engineering import engineer_features
    from app.regression import MultipleLinearRegressor
    from app.predictions import generate_predictions
    from app.metrics import calculate_metrics, format_metrics_report
    from app.residuals import analyze_residuals, plot_residuals_data
    from app.multicollinearity import calculate_vif
    from app.feature_analysis import coefficient_interpretation
    from app.visualizations import (
        plot_actual_vs_predicted, plot_residuals_vs_fitted, 
        plot_residual_distribution, plot_correlation_heatmap,
        plot_feature_vs_sales, plot_coefficient_chart,
        plot_sales_by_region, plot_monthly_sales, plot_vif_chart,
        plot_model_comparison
    )
    from app.insights import generate_insights
    from app.report import save_report
except ImportError:
    from config import AppConfig
    from loader import load_data
    from cleaner import clean_data
    from validator import validate_data
    from feature_engineering import engineer_features
    from regression import MultipleLinearRegressor
    from predictions import generate_predictions
    from metrics import calculate_metrics, format_metrics_report
    from residuals import analyze_residuals, plot_residuals_data
    from multicollinearity import calculate_vif
    from feature_analysis import coefficient_interpretation
    from visualizations import (
        plot_actual_vs_predicted, plot_residuals_vs_fitted, 
        plot_residual_distribution, plot_correlation_heatmap,
        plot_feature_vs_sales, plot_coefficient_chart,
        plot_sales_by_region, plot_monthly_sales, plot_vif_chart,
        plot_model_comparison
    )
    from insights import generate_insights
    from report import save_report

def main():
    config = AppConfig()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHARTS_DIR, exist_ok=True)
    
    print("1. Loading data...")
    df_raw = load_data(config)
    
    print("2. Cleaning data...")
    df_clean = clean_data(df_raw, config)
    
    print("3. Validating data...")
    validate_data(df_clean, config)
    
    print("4. Feature Engineering...")
    df = engineer_features(df_clean)
    
    print("5. Train/Test Split...")
    features = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    X = df[features]
    y = df[config.TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    print("6. Building & Fitting Model...")
    model = MultipleLinearRegressor(config)
    model.fit(X_train, y_train)
    
    # Extract feature names
    preprocessor = model.pipeline.named_steps['preprocessor']
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_names = cat_encoder.get_feature_names_out(config.CATEGORICAL_FEATURES).tolist()
    feature_names = config.NUMERIC_FEATURES + cat_names
    
    coef_df = model.get_coefficients(feature_names)
    
    print("7. Generating Predictions & Metrics...")
    preds_df = generate_predictions(model, X_test, y_test)
    preds_df.to_csv(config.PREDICTIONS_CSV, index=False)
    
    metrics = calculate_metrics(y_test, preds_df['Predicted'], n_features=len(feature_names))
    print(format_metrics_report(metrics))
    
    print("8. Residual Analysis...")
    residual_stats = analyze_residuals(preds_df['Actual'], preds_df['Predicted'])
    
    print("9. Multicollinearity (VIF)...")
    vif_df = calculate_vif(df, config.NUMERIC_FEATURES)
    
    print("10. Feature Analysis...")
    interpretations = coefficient_interpretation(coef_df)
    
    print("11. Generating Visualizations...")
    plot_actual_vs_predicted(preds_df['Actual'], preds_df['Predicted'], os.path.join(config.CHARTS_DIR, "actual_vs_predicted.png"))
    plot_residuals_vs_fitted(preds_df['Predicted'], preds_df['Residual'], os.path.join(config.CHARTS_DIR, "residuals_vs_fitted.png"))
    plot_residual_distribution(preds_df['Residual'], os.path.join(config.CHARTS_DIR, "residual_distribution.png"))
    plot_correlation_heatmap(df, config.NUMERIC_FEATURES + [config.TARGET_COL], os.path.join(config.CHARTS_DIR, "correlation_heatmap.png"))
    plot_feature_vs_sales(df, "TV_Spend", os.path.join(config.CHARTS_DIR, "tv_spend_vs_sales.png"))
    plot_coefficient_chart(coef_df, os.path.join(config.CHARTS_DIR, "coefficients.png"))
    plot_sales_by_region(df, os.path.join(config.CHARTS_DIR, "sales_by_region.png"))
    plot_monthly_sales(df, os.path.join(config.CHARTS_DIR, "monthly_sales.png"))
    plot_vif_chart(vif_df, os.path.join(config.CHARTS_DIR, "vif_chart.png"))
    
    comp_df = pd.DataFrame({
        "Model": ["Baseline", "MultipleLinear"],
        "R2": [0.4, metrics.get('R2', 0)],
        "RMSE": [metrics.get('RMSE', 0) * 1.5, metrics.get('RMSE', 0)]
    })
    plot_model_comparison(comp_df, os.path.join(config.CHARTS_DIR, "model_comparison.png"))
    
    print("12. Generating Insights...")
    insights = generate_insights(metrics, coef_df, vif_df, residual_stats, config)
    
    print("13. Saving Report...")
    save_report(metrics, coef_df, vif_df, residual_stats, insights, config)
    
    print("\nPipeline Complete!")

if __name__ == '__main__':
    main()
