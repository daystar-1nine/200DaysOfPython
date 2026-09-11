"""
Task 4: One-Tailed vs Two-Tailed Tests
Demonstrates how test directionality affects critical values, rejection regions, and p-values.
"""

from scipy import stats

def analyze_tails(t_stat: float, df: int, alpha: float = 0.05) -> dict:
    # Two-tailed
    t_crit_two = stats.t.ppf(1 - alpha / 2, df=df)
    p_two = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
    reject_two = abs(t_stat) >= t_crit_two
    
    # Right-tailed (greater)
    t_crit_right = stats.t.ppf(1 - alpha, df=df)
    p_right = 1 - stats.t.cdf(t_stat, df=df)
    reject_right = t_stat >= t_crit_right
    
    # Left-tailed (less)
    t_crit_left = stats.t.ppf(alpha, df=df)
    p_left = stats.t.cdf(t_stat, df=df)
    reject_left = t_stat <= t_crit_left
    
    return {
        "two_tailed": {"crit": round(t_crit_two, 4), "p_val": round(p_two, 5), "reject": reject_two},
        "right_tailed": {"crit": round(t_crit_right, 4), "p_val": round(p_right, 5), "reject": reject_right},
        "left_tailed": {"crit": round(t_crit_left, 4), "p_val": round(p_left, 5), "reject": reject_left}
    }

if __name__ == "__main__":
    t_val = 1.85
    degrees = 40
    res = analyze_tails(t_stat=t_val, df=degrees, alpha=0.05)
    print(f"=== Task 4: Tail Comparison (t = {t_val}, df = {degrees}, alpha = 0.05) ===")
    for tail, details in res.items():
        print(f"  {tail:12s} -> Critical: {details['crit']}, P-Value: {details['p_val']}, Reject: {details['reject']}")
