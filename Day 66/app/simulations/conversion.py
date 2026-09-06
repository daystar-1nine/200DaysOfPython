"""
Marketing conversion rate simulation and A/B test probability modeling.
"""
import numpy as np
import pandas as pd

def simulate_conversion_experiment(
    n_visitors_control: int = 50_000,
    n_visitors_variant: int = 50_000,
    p_control: float = 0.040,   # 4.0% baseline
    p_variant: float = 0.046,   # 4.6% variant
    seed: int = 42
) -> dict:
    """Simulate e-commerce landing page conversion A/B experiment."""
    rng = np.random.default_rng(seed)
    
    conv_control = rng.binomial(1, p_control, size=n_visitors_control)
    conv_variant = rng.binomial(1, p_variant, size=n_visitors_variant)
    
    k_control = int(np.sum(conv_control))
    k_variant = int(np.sum(conv_variant))
    
    p_hat_ctrl = k_control / n_visitors_control
    p_hat_var = k_variant / n_visitors_variant
    
    se_ctrl = np.sqrt(p_hat_ctrl * (1 - p_hat_ctrl) / n_visitors_control)
    se_var = np.sqrt(p_hat_var * (1 - p_hat_var) / n_visitors_variant)
    
    uplift = (p_hat_var - p_hat_ctrl) / p_hat_ctrl
    pooled_se = np.sqrt(se_ctrl**2 + se_var**2)
    z_score = (p_hat_var - p_hat_ctrl) / pooled_se
    
    df_summary = pd.DataFrame([
        {
            "Variant": "Control (A)",
            "Visitors": n_visitors_control,
            "Conversions": k_control,
            "True_Rate": p_control,
            "Empirical_Rate": round(p_hat_ctrl, 5),
            "Std_Error": round(se_ctrl, 5),
            "95% CI Lower": round(p_hat_ctrl - 1.96 * se_ctrl, 5),
            "95% CI Upper": round(p_hat_ctrl + 1.96 * se_ctrl, 5)
        },
        {
            "Variant": "Variant (B)",
            "Visitors": n_visitors_variant,
            "Conversions": k_variant,
            "True_Rate": p_variant,
            "Empirical_Rate": round(p_hat_var, 5),
            "Std_Error": round(se_var, 5),
            "95% CI Lower": round(p_hat_var - 1.96 * se_var, 5),
            "95% CI Upper": round(p_hat_var + 1.96 * se_var, 5)
        }
    ])
    
    return {
        "df_summary": df_summary,
        "relative_uplift_pct": round(uplift * 100, 2),
        "z_score": round(float(z_score), 3),
        "statistically_significant": bool(z_score > 1.96)
    }
