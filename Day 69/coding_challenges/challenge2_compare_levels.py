"""
Day 69 Challenge 2: Multi-Level Confidence Comparison
Computes 90%, 95%, and 99% confidence intervals for a given dataset.
"""
import numpy as np
from scipy import stats

def compare_confidence_levels(data: np.ndarray | list[float]) -> dict[str, dict]:
    """
    Calculate confidence intervals for 90%, 95%, and 99% levels on the given sample.
    """
    arr = np.asarray(data, dtype=float)
    if len(arr) < 2:
        raise ValueError("Data must contain at least 2 observations.")
        
    n = len(arr)
    df = n - 1
    mean_val = float(np.mean(arr))
    s = float(np.std(arr, ddof=1))
    se = float(s / np.sqrt(n))
    
    levels = [0.90, 0.95, 0.99]
    res = {}
    for conf in levels:
        alpha = 1.0 - conf
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
        me = float(t_crit * se)
        lbl = f"{int(conf * 100)}%"
        res[lbl] = {
            "confidence": conf,
            "critical_t": round(t_crit, 4),
            "margin_of_error": round(me, 4),
            "lower_bound": round(mean_val - me, 4),
            "upper_bound": round(mean_val + me, 4),
            "width": round(2 * me, 4)
        }
    return res

def main():
    print("=" * 70)
    print("  CHALLENGE 2: MULTI-LEVEL CONFIDENCE COMPARISON")
    print("=" * 70)
    data = [102.5, 98.4, 105.1, 99.8, 103.2, 97.9, 101.4, 104.0, 100.5, 102.1]
    comp = compare_confidence_levels(data)
    
    print(f"{'Level':<8} {'Critical t':<12} {'Margin of Error':<18} {'Lower Bound':<14} {'Upper Bound':<14} {'Width':<10}")
    print("-" * 76)
    for lvl, d in comp.items():
        print(f"{lvl:<8} {d['critical_t']:<12.4f} +/- {d['margin_of_error']:<14.4f} {d['lower_bound']:<14.4f} {d['upper_bound']:<14.4f} {d['width']:<10.4f}")

if __name__ == "__main__":
    main()
