"""
Unit tests for automated business insights generator.
"""
import pytest
from app.insights import generate_business_insights

def test_insights_generation_count():
    eda_dict = {"pearson_r": 0.85}
    train_m = {"mae": 1000.0, "rmse": 1500.0, "r2": 0.88}
    test_m = {"mae": 1100.0, "rmse": 1600.0, "r2": 0.86}
    params = {"slope": 1.85, "intercept": 25000.0}
    res_info = {"variance_ratio": 1.1, "is_homoscedastic": True}
    
    insights = generate_business_insights(
        eda_dict, train_m, test_m, params, res_info, pred_50k=117500.0
    )
    assert len(insights) >= 5

def test_insights_no_forbidden_causal_language():
    eda_dict = {"pearson_r": 0.85}
    train_m = {"mae": 1000.0, "rmse": 1500.0, "r2": 0.88}
    test_m = {"mae": 1100.0, "rmse": 1600.0, "r2": 0.86}
    params = {"slope": 1.85, "intercept": 25000.0}
    res_info = {"variance_ratio": 1.1, "is_homoscedastic": True}
    
    insights = generate_business_insights(
        eda_dict, train_m, test_m, params, res_info, pred_50k=117500.0
    )
    forbidden = ["causes", "cause to", "triggers", "forces", "leads directly to"]
    for text in insights:
        for word in forbidden:
            assert word not in text.lower(), f"Forbidden causal word '{word}' found in insight!"

def test_insights_mandate_present():
    eda_dict = {"pearson_r": 0.85}
    train_m = {"mae": 1000.0, "rmse": 1500.0, "r2": 0.88}
    test_m = {"mae": 1100.0, "rmse": 1600.0, "r2": 0.86}
    params = {"slope": 1.85, "intercept": 25000.0}
    res_info = {"variance_ratio": 1.1, "is_homoscedastic": True}
    
    insights = generate_business_insights(
        eda_dict, train_m, test_m, params, res_info, pred_50k=117500.0
    )
    assert any("methodological mandate" in text.lower() for text in insights)

def test_insights_extrapolation_warning():
    eda_dict = {"pearson_r": 0.85}
    train_m = {"mae": 1000.0, "rmse": 1500.0, "r2": 0.88}
    test_m = {"mae": 1100.0, "rmse": 1600.0, "r2": 0.86}
    params = {"slope": 1.85, "intercept": 25000.0}
    res_info = {"variance_ratio": 1.1, "is_homoscedastic": True}
    
    insights = generate_business_insights(
        eda_dict, train_m, test_m, params, res_info, pred_50k=117500.0
    )
    assert any("extrapolation" in text.lower() for text in insights)
