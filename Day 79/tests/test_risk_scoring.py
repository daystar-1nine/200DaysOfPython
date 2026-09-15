import numpy as np
import pandas as pd
from app.risk_scoring import assign_risk_tiers

def test_risk_tiers_categorization():
    probs = np.array([0.10, 0.35, 0.65, 0.85])
    df_risk = assign_risk_tiers(probs)
    assert list(df_risk['Risk_Tier']) == ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']

def test_risk_tiers_urgency():
    probs = np.array([0.10, 0.35, 0.65, 0.85])
    df_risk = assign_risk_tiers(probs)
    assert list(df_risk['Urgency']) == ['None', 'Low', 'Medium', 'Urgent']

def test_risk_tiers_length():
    probs = np.random.uniform(0, 1, 50)
    df_risk = assign_risk_tiers(probs)
    assert len(df_risk) == 50

def test_risk_boundary_conditions():
    probs = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    df_risk = assign_risk_tiers(probs)
    # 0.0 -> Low, 0.25 -> Medium, 0.50 -> High, 0.75 -> Critical, 1.0 -> Critical
    assert df_risk.loc[0, 'Risk_Tier'] == 'Low Risk'
    assert df_risk.loc[1, 'Risk_Tier'] == 'Medium Risk'
    assert df_risk.loc[2, 'Risk_Tier'] == 'High Risk'
    assert df_risk.loc[3, 'Risk_Tier'] == 'Critical Risk'
    assert df_risk.loc[4, 'Risk_Tier'] == 'Critical Risk'
