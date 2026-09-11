"""
Experimentation metric calculations: conversion, lifts, ARPU, AOV, and guardrails.
"""

import math
import numpy as np
import pandas as pd
from scipy import stats

def compute_conversion_metrics(df_ctrl: pd.DataFrame, df_trt: pd.DataFrame) -> dict:
    n_c = len(df_ctrl)
    n_t = len(df_trt)
    conv_c = int(df_ctrl["converted"].sum())
    conv_t = int(df_trt["converted"].sum())
    
    rate_c = conv_c / n_c if n_c > 0 else 0.0
    rate_t = conv_t / n_t if n_t > 0 else 0.0
    abs_lift = rate_t - rate_c
    rel_lift = (rate_t - rate_c) / rate_c * 100.0 if rate_c > 0 else float("inf")
    
    return {
        "n_control": n_c,
        "n_treatment": n_t,
        "conversions_control": conv_c,
        "conversions_treatment": conv_t,
        "rate_control": rate_c,
        "rate_treatment": rate_t,
        "absolute_lift": abs_lift,
        "relative_lift_pct": rel_lift
    }

def compute_financial_metrics(df_ctrl: pd.DataFrame, df_trt: pd.DataFrame) -> dict:
    rev_c = df_ctrl["revenue"].values
    rev_t = df_trt["revenue"].values
    
    tot_rev_c = float(np.sum(rev_c))
    tot_rev_t = float(np.sum(rev_t))
    
    arpu_c = float(np.mean(rev_c)) if len(rev_c) > 0 else 0.0
    arpu_t = float(np.mean(rev_t)) if len(rev_t) > 0 else 0.0
    
    conv_rev_c = rev_c[rev_c > 0]
    conv_rev_t = rev_t[rev_t > 0]
    aov_c = float(np.mean(conv_rev_c)) if len(conv_rev_c) > 0 else 0.0
    aov_t = float(np.mean(conv_rev_t)) if len(conv_rev_t) > 0 else 0.0
    
    return {
        "total_revenue_control": round(tot_rev_c, 2),
        "total_revenue_treatment": round(tot_rev_t, 2),
        "arpu_control": round(arpu_c, 2),
        "arpu_treatment": round(arpu_t, 2),
        "arpu_lift": round(arpu_t - arpu_c, 2),
        "aov_control": round(aov_c, 2),
        "aov_treatment": round(aov_t, 2),
        "aov_lift": round(aov_t - aov_c, 2)
    }

def compute_guardrail_metrics(df_ctrl: pd.DataFrame, df_trt: pd.DataFrame) -> dict:
    bounce_c = float(df_ctrl["bounce"].mean())
    bounce_t = float(df_trt["bounce"].mean())
    
    refund_c = float(df_ctrl["refunded"].mean())
    refund_t = float(df_trt["refunded"].mean())
    
    dur_c = float(df_ctrl["session_duration"].mean())
    dur_t = float(df_trt["session_duration"].mean())
    
    return {
        "bounce_rate_control": round(bounce_c, 4),
        "bounce_rate_treatment": round(bounce_t, 4),
        "bounce_delta": round(bounce_t - bounce_c, 4),
        "refund_rate_control": round(refund_c, 4),
        "refund_rate_treatment": round(refund_t, 4),
        "refund_delta": round(refund_t - refund_c, 4),
        "session_duration_control": round(dur_c, 1),
        "session_duration_treatment": round(dur_t, 1),
        "duration_delta": round(dur_t - dur_c, 1)
    }

def check_sample_ratio_mismatch(n_control: int, n_treatment: int, expected_ratio: float = 0.50) -> dict:
    total = n_control + n_treatment
    if total == 0:
        return {"srm_detected": False, "p_value": 1.0}
        
    expected_c = total * expected_ratio
    expected_t = total * (1.0 - expected_ratio)
    
    # Chi-square goodness-of-fit test
    chi2_stat = ((n_control - expected_c)**2 / expected_c) + ((n_treatment - expected_t)**2 / expected_t)
    p_val = float(1.0 - stats.chi2.cdf(chi2_stat, df=1))
    
    # SRM is flagged if p < 0.001
    is_srm = p_val < 0.001
    return {
        "observed_control": n_control,
        "observed_treatment": n_treatment,
        "expected_control": expected_c,
        "expected_treatment": expected_t,
        "chi2_statistic": round(chi2_stat, 4),
        "p_value": p_val,
        "srm_detected": is_srm
    }
