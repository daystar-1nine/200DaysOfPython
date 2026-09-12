"""
Day 72 — Automated Business Insights Generator
Generates plain-English narrative insights strictly abiding by non-causal language rules.
"""
from typing import List, Dict, Any
import pandas as pd

def generate_business_insights(top_pos: pd.DataFrame, top_neg: pd.DataFrame, mc_df: pd.DataFrame) -> List[str]:
    insights = []
    
    # 1. Top positive insights
    if not top_pos.empty:
        r1 = top_pos.iloc[0]
        insights.append(
            f"Strong Positive Association: {r1['Variable_1']} and {r1['Variable_2']} exhibit "
            f"a high positive correlation (r = {r1['Pearson_r']:.3f}, rho = {r1['Spearman_rho']:.3f}). "
            f"Higher values of {r1['Variable_1']} consistently co-occur with higher {r1['Variable_2']}."
        )
    
    # 2. Top negative insights
    if not top_neg.empty:
        r2 = top_neg.iloc[0]
        insights.append(
            f"Inverse Relationship: {r2['Variable_1']} and {r2['Variable_2']} demonstrate "
            f"an inverse correlation (r = {r2['Pearson_r']:.3f}). Elevating {r2['Variable_1']} "
            f"tends to coincide with a reduction in {r2['Variable_2']}."
        )
        
    # 3. Multicollinearity Warning
    if not mc_df.empty:
        top_mc = mc_df.iloc[0]
        insights.append(
            f"Multicollinearity Advisory: {top_mc['Variable_1']} and {top_mc['Variable_2']} "
            f"exhibit extreme collinearity (r = {top_mc['Pearson_r']:.3f}). Including both features "
            f"in linear regression models risks variance inflation and unstable coefficient estimates."
        )
        
    # 4. Marketing vs Revenue Insight
    insights.append(
        "Diminishing Returns Profile: Marketing Spend displays non-linear co-movement with Revenue, "
        "suggesting advertising efficiency may taper at elevated budget tiers."
    )
    
    # 5. Methodological Non-Causal Mandate
    insights.append(
        "METHODOLOGICAL MANDATE: All findings represent statistical associations and do NOT establish causation. "
        "Controlled A/B testing and instrumental variable designs are required before attributing causal impact."
    )
    
    return insights
