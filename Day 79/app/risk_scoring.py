from typing import List
import numpy as np
import pandas as pd

def assign_risk_tiers(churn_probabilities: np.ndarray) -> pd.DataFrame:
    """Assign churn risk tier and recommended CRM action based on predicted probability."""
    tiers = []
    actions = []
    urgencies = []
    
    for prob in churn_probabilities:
        if prob < 0.25:
            tiers.append('Low Risk')
            actions.append('Standard engagement & loyalty perks')
            urgencies.append('None')
        elif prob < 0.50:
            tiers.append('Medium Risk')
            actions.append('Automated satisfaction check-in & feature updates')
            urgencies.append('Low')
        elif prob < 0.75:
            tiers.append('High Risk')
            actions.append('Proactive outreach & discounted renewal offer')
            urgencies.append('Medium')
        else:
            tiers.append('Critical Risk')
            actions.append('Immediate senior retention agent escalation')
            urgencies.append('Urgent')
            
    return pd.DataFrame({
        'Churn_Probability': churn_probabilities,
        'Risk_Tier': tiers,
        'Action': actions,
        'Urgency': urgencies
    })
