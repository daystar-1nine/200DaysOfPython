"""
Challenge 3: Comprehensive A/B Test Reporting Engine
Ingests Control and Treatment DataFrames containing user activity and generates
a structured multi-metric report across primary conversion, financial revenue, and guardrails.
"""

import math
import pandas as pd
import numpy as np
from scipy import stats

def ab_test_report(df_control: pd.DataFrame, df_treatment: pd.DataFrame, alpha: float = 0.05) -> dict:
    n_c = len(df_control)
    n_t = len(df_treatment)
    if n_c < 10 or n_t < 10:
        raise ValueError("Both groups must have at least 10 observations.")
        
    # 1. Conversion Metrics
    conv_c = int(df_control["converted"].sum())
    conv_t = int(df_treatment["converted"].sum())
    rate_c = conv_c / n_c
    rate_t = conv_t / n_t
    abs_lift = rate_t - rate_c
    rel_lift = (rate_t - rate_c) / rate_c * 100.0 if rate_c > 0 else 0.0
    
    p_pool = (conv_c + conv_t) / (n_c + n_t)
    se_pool = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_c + 1.0 / n_t))
    z_conv = abs_lift / se_pool if se_pool > 0 else 0.0
    p_conv = float(2 * (1 - stats.norm.cdf(abs(z_conv))))
    
    # 2. Revenue Metrics (ARPU across all users)
    arpu_c = float(df_control["revenue"].mean())
    arpu_t = float(df_treatment["revenue"].mean())
    arpu_diff = arpu_t - arpu_c
    t_rev, p_rev = stats.ttest_ind(df_treatment["revenue"], df_control["revenue"], equal_var=False)
    
    # 3. Guardrail Metrics
    bounce_c = float(df_control["bounce"].mean())
    bounce_t = float(df_treatment["bounce"].mean())
    refund_c = float(df_control["refunded"].mean())
    refund_t = float(df_treatment["refunded"].mean())
    dur_c = float(df_control["session_duration"].mean())
    dur_t = float(df_treatment["session_duration"].mean())
    
    # Guardrail checks: bounce rate not degraded by > 5 pp, refund rate not degraded by > 1.5 pp
    guardrails_ok = (bounce_t - bounce_c <= 0.05) and (refund_t - refund_c <= 0.015)
    
    # Recommendation logic
    if p_conv < alpha and abs_lift > 0 and guardrails_ok:
        recommendation = "RECOMMEND TREATMENT LAUNCH (Statistically Significant Lift & Guardrails Healthy)"
    elif p_conv < alpha and abs_lift < 0:
        recommendation = "DO NOT LAUNCH (Statistically Significant Negative Regression)"
    elif not guardrails_ok:
        recommendation = "REJECT TREATMENT (Guardrail Metrics Breached)"
    else:
        recommendation = "INCONCLUSIVE (Insufficient Evidence of Uplift; Continue Testing)"
        
    return {
        "sample": {"control_users": n_c, "treatment_users": n_t, "total_users": n_c + n_t},
        "conversion": {
            "control_rate": round(rate_c, 4),
            "treatment_rate": round(rate_t, 4),
            "absolute_lift": round(abs_lift, 4),
            "relative_lift_pct": round(rel_lift, 2),
            "z_statistic": round(z_conv, 4),
            "p_value": p_conv,
            "statistically_significant": bool(p_conv < alpha)
        },
        "revenue": {
            "control_arpu": round(arpu_c, 2),
            "treatment_arpu": round(arpu_t, 2),
            "arpu_lift": round(arpu_diff, 2),
            "t_statistic": round(float(t_rev), 4),
            "p_value": float(p_rev)
        },
        "guardrails": {
            "control_bounce_rate": round(bounce_c, 4),
            "treatment_bounce_rate": round(bounce_t, 4),
            "control_refund_rate": round(refund_c, 4),
            "treatment_refund_rate": round(refund_t, 4),
            "control_session_duration": round(dur_c, 1),
            "treatment_session_duration": round(dur_t, 1),
            "guardrails_satisfied": guardrails_ok
        },
        "final_recommendation": recommendation
    }

if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df_data = pd.read_csv(os.path.join(base, "data", "experiment_users.csv"))
    ctrl = df_data[df_data["group"] == "Control"]
    trt = df_data[df_data["group"] == "Treatment"]
    rep = ab_test_report(ctrl, trt)
    print("=== Challenge 3: ab_test_report Output ===")
    print("  Sample:", rep["sample"])
    print("  Conversion:", rep["conversion"])
    print("  Revenue:", rep["revenue"])
    print("  Guardrails:", rep["guardrails"])
    print("  Recommendation:", rep["final_recommendation"])
