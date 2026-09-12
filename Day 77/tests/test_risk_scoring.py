import numpy as np
from app.risk_scoring import score_customers

def test_score_customers():
    cids = ["C1", "C2", "C3"]
    probs = np.array([0.15, 0.45, 0.85])
    df_scores = score_customers(cids, probs)
    assert len(df_scores) == 3
    assert list(df_scores["Risk_Level"]) == ["Low", "Medium", "High"]
    assert "Recommended_Action" in df_scores.columns
    assert (df_scores["Risk_Score"] >= 0).all() and (df_scores["Risk_Score"] <= 100).all()
