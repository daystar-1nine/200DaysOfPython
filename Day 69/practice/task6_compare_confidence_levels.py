"""
Day 69 Task 6: Confidence Level Comparison
Demonstrates the precision-confidence tradeoff across 90%, 95%, and 99% levels.
"""
import numpy as np
from scipy import stats

def compare_confidence_levels(data: np.ndarray, levels: list[float] = [0.90, 0.95, 0.99]) -> list[dict]:
    n = len(data)
    df = n - 1
    mean_val = float(np.mean(data))
    s = float(np.std(data, ddof=1))
    se = s / np.sqrt(n)
    
    results = []
    for conf in levels:
        alpha = 1.0 - conf
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
        me = t_crit * se
        results.append({
            "confidence_level": f"{conf*100:.0f}%",
            "critical_t": round(t_crit, 4),
            "margin_of_error": round(me, 4),
            "lower_bound": round(mean_val - me, 4),
            "upper_bound": round(mean_val + me, 4),
            "interval_width": round(2 * me, 4)
        })
    return results

def main():
    print("=" * 70)
    print("  TASK 6: CONFIDENCE LEVEL COMPARISON (90% vs 95% vs 99%)")
    print("=" * 70)
    data = np.array([72.0, 74.0, 68.0, 75.0, 71.0, 69.0, 73.0, 70.0, 76.0, 67.0, 73.0, 72.0])
    comps = compare_confidence_levels(data)
    
    print(f"{'Level':<8} {'Critical t':<12} {'Margin of Error':<18} {'Lower Bound':<14} {'Upper Bound':<14} {'Width':<10}")
    print("-" * 76)
    for c in comps:
        print(f"{c['confidence_level']:<8} {c['critical_t']:<12.4f} +/- {c['margin_of_error']:<14.4f} {c['lower_bound']:<14.4f} {c['upper_bound']:<14.4f} {c['interval_width']:<10.4f}")
    print("-" * 76)
    print("Insight: Higher confidence requires wider intervals (the precision tax).")

if __name__ == "__main__":
    main()
