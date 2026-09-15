import numpy as np
from app.risk_scoring import score_customers

def test_score_customers():
    cids = ["C1", "C2", "C3"]
    probs = np.array([0.15, 0.45, 0.85])
    df_s = score_customers(cids, probs)
    assert len(df_s) == 3
    assert list(df_s["Risk_Level"]) == ["Low", "Medium", "High"]
    assert "Recommended_Action" in df_s.columns
    assert (df_s["Risk_Score"] >= 0).all() and (df_s["Risk_Score"] <= 100).all()
