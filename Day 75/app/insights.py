import pandas as pd

def generate_insights(comparison_df, best_model_name, tuning_results, residual_stats, coef_df) -> list:
    insights = []
    insights.append(f"The best performing model is {best_model_name}.")
    
    overfit = comparison_df.apply(lambda row: (row['Test_RMSE'] - row['Train_RMSE'])/row['Train_RMSE'] > 0.15, axis=1)
    if overfit.any():
        insights.append("Some models exhibit signs of overfitting, where test error is significantly higher than training error.")
    
    if not coef_df.empty:
        top_feature = coef_df['Feature'].iloc[0]
        insights.append(f"The feature most strongly associated with the target is {top_feature}.")
        
    insights.append("Regularization (Ridge/Lasso) helps in reducing model variance for complex polynomial models.")
    insights.append("Linear models provide a good baseline, but polynomial terms capture non-linear relationships.")
    insights.append(f"Residual analysis shows a mean error of {residual_stats['mean']:.4f}.")
    if abs(residual_stats['skewness']) > 1:
        insights.append("Residuals are skewed, indicating potential outliers or missed non-linear patterns.")
        
    insights.append(f"Hyperparameter tuning selected degree {tuning_results['polynomial']['best_degree']} for polynomial regression.")
    insights.append("L1 regularization (Lasso) can be used for feature selection by driving some coefficients to zero.")
    insights.append("L2 regularization (Ridge) prevents large coefficient values, improving generalization.")
    
    return insights
