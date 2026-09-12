import numpy as np
import pandas as pd

def score_customers(customer_ids, churn_probs, risk_preds=None) -> pd.DataFrame:
    records = []
    for i, cid in enumerate(customer_ids):
        p = float(churn_probs[i])
        
        # Derive risk level if not explicitly provided
        if risk_preds is not None:
            r_level = str(risk_preds[i])
        else:
            if p < 0.30:
                r_level = "Low"
            elif p < 0.65:
                r_level = "Medium"
            else:
                r_level = "High"
                
        # Prescribed business recommendation
        if r_level == "High" or p >= 0.65:
            action = "Priority retention outreach and contract renewal incentive"
        elif r_level == "Medium" or p >= 0.30:
            action = "Monitor usage patterns and deploy targeted value engagement"
        else:
            action = "Standard loyalty engagement and service maintenance"
            
        records.append({
            "Customer_ID": cid,
            "Churn_Probability": round(p, 4),
            "Risk_Score": round(p * 100, 1),
            "Risk_Level": r_level,
            "Recommended_Action": action
        })
        
    return pd.DataFrame(records)
