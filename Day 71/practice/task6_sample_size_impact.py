"""
Task 6: Sample Size Scaling in A/B Testing
Evaluates the effect of increasing sample size from 1,000 to 100,000
on Standard Error, Confidence Interval width, and p-value.
"""

import math
from scipy import stats

def evaluate_sample_size_impact():
    p_a = 0.052
    p_b = 0.058
    true_diff = p_b - p_a
    sample_sizes = [1000, 5000, 10000, 50000, 100000]
    
    print("=== Task 6: Sample Size Impact on Precision and Significance ===")
    print(f"{'Sample Size (N)':>15} | {'Std Error':>10} | {'95% CI Width':>14} | {'z-stat':>8} | {'p-value':>12} | {'Reject H0?':>10}")
    print("-" * 80)
    
    for n in sample_sizes:
        conv_a = int(round(n * p_a))
        conv_b = int(round(n * p_b))
        
        # Pooled test
        p_pool = (conv_a + conv_b) / (2 * n)
        se = math.sqrt(p_pool * (1.0 - p_pool) * (2.0 / n))
        z = true_diff / se
        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
        
        # CI width
        ci_width = 2.0 * 1.96 * se
        reject = "REJECT" if p_val < 0.05 else "FAIL-REJ"
        
        print(f"{n:>15,d} | {se:>10.6f} | {ci_width:>14.6f} | {z:>8.2f} | {p_val:>12.4e} | {reject:>10}")

if __name__ == "__main__":
    evaluate_sample_size_impact()
