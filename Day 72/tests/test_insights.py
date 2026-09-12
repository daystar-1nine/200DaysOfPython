"""
Unit tests for automated business insights and causal governance.
"""
import pytest
import pandas as pd
from app.insights import generate_business_insights

def test_insights_no_forbidden_causal_language():
    top_pos = pd.DataFrame([{
        "Variable_1": "Revenue", "Variable_2": "Profit",
        "Pearson_r": 0.85, "Spearman_rho": 0.87
    }])
    top_neg = pd.DataFrame([{
        "Variable_1": "Discount", "Variable_2": "Profit",
        "Pearson_r": -0.45, "Spearman_rho": -0.48
    }])
    mc_df = pd.DataFrame([{
        "Variable_1": "Cost_Price", "Variable_2": "Unit_Price",
        "Pearson_r": 0.94
    }])
    
    insights = generate_business_insights(top_pos, top_neg, mc_df)
    assert len(insights) >= 4
    
    forbidden_words = ["causes", "cause to", "triggers", "forces", "leads directly to"]
    for text in insights:
        for word in forbidden_words:
            assert word not in text.lower(), f"Forbidden causal word '{word}' found in insight!"
            
    # Mandate check
    mandate_present = any("methodological mandate" in t.lower() for t in insights)
    assert mandate_present, "Expected non-causal methodological mandate in insights."
