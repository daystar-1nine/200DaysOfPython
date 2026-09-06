"""
Day 67 Task 1: Bernoulli Distribution Calculations
Models a customer conversion event with P(success) = 0.7,
computing PMF, Expected Value, and Variance.
"""
from scipy.stats import bernoulli

def main():
    p = 0.7
    rv = bernoulli(p)
    
    p_success = float(rv.pmf(1))
    p_failure = float(rv.pmf(0))
    expected_value = float(rv.mean())
    variance = float(rv.var())
    std_dev = float(rv.std())
    
    print("=" * 60)
    print("  TASK 1: BERNOULLI DISTRIBUTION (p = 0.7)")
    print("=" * 60)
    print(f"P(Success) [X = 1]: {p_success:.4f}")
    print(f"P(Failure) [X = 0]: {p_failure:.4f}")
    print(f"Expected Value:     {expected_value:.4f}")
    print(f"Variance:           {variance:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")

if __name__ == "__main__":
    main()
