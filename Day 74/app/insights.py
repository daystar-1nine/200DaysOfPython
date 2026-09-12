import pandas as pd

try:
    from app.config import AppConfig
except ImportError:
    from config import AppConfig

def generate_business_insights(coef_df: pd.DataFrame) -> list[str]:
    """
    Generates non-causal observational insights from model coefficients.
    """
    insights = []
    if coef_df is None or coef_df.empty:
        return ["No coefficient data available for generating insights."]

    pos = coef_df[coef_df['Coefficient'] > 0]
    neg = coef_df[coef_df['Coefficient'] < 0]

    if not pos.empty:
        top_pos = pos.sort_values(by='Coefficient', ascending=False).iloc[0]
        insights.append(
            f"Feature '{top_pos['Feature']}' exhibits the strongest positive association with target values "
            f"(coefficient = {top_pos['Coefficient']:.3f}), holding other predictors constant."
        )

    if not neg.empty:
        top_neg = neg.sort_values(by='Coefficient', ascending=True).iloc[0]
        insights.append(
            f"Feature '{top_neg['Feature']}' exhibits an inverse estimated relationship with target values "
            f"(coefficient = {top_neg['Coefficient']:.3f}), holding other predictors constant."
        )

    insights.append(
        "Estimated regression weights reflect correlational associations under the specified model structure, "
        "not direct causal mechanisms."
    )
    insights.append(
        f"A total of {len(coef_df)} predictive variables were evaluated across the linear specification."
    )
    return insights

def generate_insights(metrics: dict, coef_df: pd.DataFrame, vif_df: pd.DataFrame, residual_stats: dict, config: AppConfig) -> list[str]:
    insights = generate_business_insights(coef_df)
    
    r2 = metrics.get('R2', 0)
    insights.append(f"Model Explanatory Power: The model accounts for roughly {r2*100:.1f}% of the variance in Sales.")
    
    threshold = getattr(config, 'VIF_THRESHOLD', 10.0)
    high_vif = vif_df[vif_df['VIF'] > threshold] if (vif_df is not None and not vif_df.empty) else pd.DataFrame()
    if not high_vif.empty:
        feats = ", ".join(high_vif['Feature'].tolist())
        insights.append(f"Multicollinearity Warning: High VIF observed for {feats}. Consider feature selection.")
    else:
        insights.append("Multicollinearity: VIF values are within acceptable bounds, suggesting independent features.")
        
    if residual_stats.get('is_zero_mean', False):
        insights.append("Residuals: Mean of residuals is close to zero, meeting a key assumption of linear regression.")
    else:
        insights.append("Residuals: Mean of residuals deviates from zero, indicating potential bias.")
        
    insights.append(f"Average Error: On average, the predictions deviate from actual Sales by roughly {metrics.get('MAE', 0):.2f}.")
    
    insights.append("Actionable Takeaway: Focus resource allocation on features with the highest positive coefficient weights.")
    insights.append("Actionable Takeaway: Review features with negative coefficients to evaluate operational trade-offs.")
    
    return insights
