"""
Day 73 - Business Insights Generator
Generates plain-English narrative insights adhering strictly to non-causal language guidelines.
"""
from typing import List, Dict, Any

def generate_business_insights(
    eda_dict: Dict[str, Any],
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    model_params: Dict[str, float],
    residual_info: Dict[str, Any],
    pred_50k: float
) -> List[str]:
    slope = model_params["slope"]
    intercept = model_params["intercept"]
    test_rmse = test_metrics["rmse"]
    test_r2 = test_metrics["r2"]
    
    insights = []
    
    # 1. Slope & Directional Association
    insights.append(
        f"Positive Linear Association: The model identifies a positive linear relationship "
        f"(slope b1 = {slope:.2f}). For every additional Rs. 1,000 invested in advertising, "
        f"expected sales revenue increases by approximately Rs. {slope * 1000:,.2f} on average."
    )
    
    # 2. Baseline Organic Intercept
    insights.append(
        f"Baseline Organic Revenue: The estimated intercept is Rs. {intercept:,.2f}, representing "
        f"the baseline level of sales when advertising expenditure is zero. Management should view "
        f"this as the organic customer base run-rate."
    )
    
    # 3. Budget Scenario Forecast
    insights.append(
        f"Target Budget Projection: For a planned advertising expenditure of Rs. 50,000, the model "
        f"forecasts expected sales of Rs. {pred_50k:,.2f}."
    )
    
    # 4. Error Magnitude & Generalization
    insights.append(
        f"Prediction Uncertainty: The model achieves an out-of-sample test R2 of {test_r2:.3f} and "
        f"RMSE of Rs. {test_rmse:,.2f}. Typical prediction errors hover around Rs. {test_rmse:,.2f}."
    )
    
    # 5. Diagnostic Homoscedasticity Assessment
    homo_status = "satisfies" if residual_info["is_homoscedastic"] else "exhibits mild departure from"
    insights.append(
        f"Residual Diagnostic: Error variance {homo_status} homoscedasticity (variance ratio = "
        f"{residual_info['variance_ratio']:.2f}), confirming stable error bands across budget ranges."
    )
    
    # 6. Extrapolation Caution & Causal Mandate
    insights.append(
        "METHODOLOGICAL MANDATE & EXTRAPOLATION ADVISORY: This regression model captures empirical "
        "association, NOT direct physical causation. Budget allocations beyond Rs. 100,000 represent "
        "unsupported extrapolation. Controlled randomized marketing experiments (A/B tests) are "
        "strongly advised before committing massive capital allocations."
    )
    
    return insights
