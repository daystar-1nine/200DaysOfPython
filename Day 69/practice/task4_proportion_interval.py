"""
Day 69 Task 4: Confidence Interval for a Population Proportion
Implements calculate_proportion_interval(successes, sample_size, confidence)
supporting both Wald Normal Approximation and Wilson Score intervals.
"""
import numpy as np
from scipy import stats

def calculate_proportion_interval(
    successes: int,
    sample_size: int,
    confidence: float = 0.95,
    method: str = "wilson"
) -> dict:
    """
    Calculate confidence interval for a population proportion.
    method: 'wald' or 'wilson'
    """
    if sample_size <= 0:
        raise ValueError("Sample size must be strictly positive.")
    if successes < 0 or successes > sample_size:
        raise ValueError(f"Successes ({successes}) must be between 0 and sample size ({sample_size}).")
    if not (0.0 < confidence < 1.0):
        raise ValueError("Confidence level must be strictly between 0 and 1.")
        
    p_hat = successes / sample_size
    alpha = 1.0 - confidence
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    
    if method == "wald":
        se = np.sqrt(p_hat * (1.0 - p_hat) / sample_size)
        me = z_crit * se
        lower = max(0.0, p_hat - me)
        upper = min(1.0, p_hat + me)
    elif method == "wilson":
        denom = 1.0 + (z_crit**2) / sample_size
        center = (p_hat + (z_crit**2) / (2.0 * sample_size)) / denom
        spread = (z_crit * np.sqrt((p_hat * (1.0 - p_hat) / sample_size) + (z_crit**2) / (4.0 * sample_size**2))) / denom
        lower = max(0.0, center - spread)
        upper = min(1.0, center + spread)
        se = np.sqrt(p_hat * (1.0 - p_hat) / sample_size)
        me = spread
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'wald' or 'wilson'.")
        
    return {
        "successes": successes,
        "sample_size": sample_size,
        "sample_proportion": round(p_hat, 4),
        "method": method,
        "standard_error": round(float(se), 4),
        "margin_of_error": round(float(me), 4),
        "lower_bound": round(float(lower), 4),
        "upper_bound": round(float(upper), 4)
    }

def main():
    print("=" * 60)
    print("  TASK 4: PROPORTION CONFIDENCE INTERVAL")
    print("=" * 60)
    n = 500
    x = 320
    conf = 0.95
    
    wald_res = calculate_proportion_interval(x, n, conf, method="wald")
    wilson_res = calculate_proportion_interval(x, n, conf, method="wilson")
    
    print(f"Survey Data: {x} satisfied out of {n} customers (p_hat = {wald_res['sample_proportion']:.1%})")
    print(f"Wald 95% CI:   [{wald_res['lower_bound']:.4f}, {wald_res['upper_bound']:.4f}] (ME: +/- {wald_res['margin_of_error']:.4f})")
    print(f"Wilson 95% CI: [{wilson_res['lower_bound']:.4f}, {wilson_res['upper_bound']:.4f}] (ME: +/- {wilson_res['margin_of_error']:.4f})")

if __name__ == "__main__":
    main()
