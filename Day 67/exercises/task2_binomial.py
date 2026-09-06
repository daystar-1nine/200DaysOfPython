"""
Day 67 Task 2: Binomial Customer Conversion Calculations
Analyzes 20 contacted customers with conversion probability p = 0.1,
evaluating exact, cumulative, survival probabilities, and moments.
"""
from scipy.stats import binom

def main():
    n = 20
    p = 0.1
    rv = binom(n, p)
    
    p_exact_3 = float(rv.pmf(3))
    p_at_most_3 = float(rv.cdf(3))
    p_at_least_3 = float(1.0 - rv.cdf(2))  # 1 - P(X <= 2)
    ev = float(rv.mean())
    variance = float(rv.var())
    std_dev = float(rv.std())
    
    print("=" * 60)
    print("  TASK 2: BINOMIAL CONVERSION (n = 20, p = 0.1)")
    print("=" * 60)
    print(f"P(Exactly 3 conversions) [P(X=3)]:  {p_exact_3:.5f} ({p_exact_3*100:.3f}%)")
    print(f"P(At most 3 conversions) [P(X<=3)]: {p_at_most_3:.5f} ({p_at_most_3*100:.3f}%)")
    print(f"P(At least 3 conversions) [P(X>=3)]:{p_at_least_3:.5f} ({p_at_least_3*100:.3f}%)")
    print(f"Expected Conversions (E[X] = np):   {ev:.2f}")
    print(f"Variance (np(1-p)):                 {variance:.2f}")
    print(f"Standard Deviation:                 {std_dev:.4f}")

if __name__ == "__main__":
    main()
