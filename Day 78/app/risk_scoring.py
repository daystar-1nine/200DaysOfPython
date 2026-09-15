import pandas as pd

def score_customers(customer_ids, churn_probs, preds=None) -> pd.DataFrame:
    records = []
    for i, cid in enumerate(customer_ids):
        p = float(churn_probs[i])
        c = int(preds[i]) if preds is not None else int(p >= 0.50)
        
        if p >= 0.65:
            r_level = "High"
            action = "Immediate proactive retention call with annual renewal discount"
        elif p >= 0.30:
            r_level = "Medium"
            action = "Engagement email sequence and service satisfaction survey"
        else:
            r_level = "Low"
            action = "Standard loyalty rewards and product recommendations"
            
        records.append({
            "Customer_ID": cid,
            "Predicted_Churn": c,
            "Churn_Probability": round(p, 4),
            "Risk_Score": round(p * 100, 1),
            "Risk_Level": r_level,
            "Recommended_Action": action
        })
    return pd.DataFrame(records)
