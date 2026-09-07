"""
Day 69 Task 7: Sample Size Impact on Confidence Interval Width
Simulates confidence intervals across n = 10, 30, 50, 100, 500 to demonstrate 1/sqrt(n) contraction.
"""
import numpy as np
from scipy import stats

def evaluate_sample_size_impact(pop_std: float = 20.0, sample_sizes: list[int] = [10, 30, 50, 100, 500], confidence: float = 0.95) -> list[dict]:
    results = []
    alpha = 1.0 - confidence
    
    for n in sample_sizes:
        df = n - 1
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
        se = pop_std / np.sqrt(n)
        me = t_crit * se
        results.append({
            "sample_size": n,
            "critical_t": round(t_crit, 4),
            "standard_error": round(se, 4),
            "margin_of_error": round(me, 4),
            "interval_width": round(2 * me, 4)
        })
    return results

def main():
    print("=" * 70)
    print("  TASK 7: SAMPLE SIZE IMPACT ON CONFIDENCE INTERVAL WIDTH")
    print("=" * 70)
    rows = evaluate_sample_size_impact(pop_std=25.0)
    print(f"{'Sample Size (n)':<18} {'Critical t':<12} {'Standard Error':<16} {'Margin of Error':<18} {'Interval Width':<14}")
    print("-" * 78)
    for r in rows:
        print(f"{r['sample_size']:<18} {r['critical_t']:<12.4f} {r['standard_error']:<16.4f} +/- {r['margin_of_error']:<14.4f} {r['interval_width']:<14.4f}")
    print("-" * 78)
    print("Insight: Increasing n from 10 to 500 dramatically shrinks interval width from ~35 to ~4.4.")

if __name__ == "__main__":
    main()
