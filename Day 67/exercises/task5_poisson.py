"""
Day 67 Task 5: Poisson Web Server Request Probabilities
Models web requests arriving at an average of lambda = 5 requests/second.
"""
from scipy.stats import poisson

def main():
    rate = 5.0
    rv = poisson(mu=rate)
    
    p_exact_3 = float(rv.pmf(3))
    p_at_most_3 = float(rv.cdf(3))
    p_more_than_5 = float(1.0 - rv.cdf(5))
    
    print("=" * 60)
    print("  TASK 5: POISSON DISTRIBUTION (lambda = 5 req/s)")
    print("=" * 60)
    print(f"P(Exactly 3 requests) [P(X=3)]:  {p_exact_3:.5f} ({p_exact_3*100:.2f}%)")
    print(f"P(At most 3 requests) [P(X<=3)]: {p_at_most_3:.5f} ({p_at_most_3*100:.2f}%)")
    print(f"P(More than 5 requests) [P(X>5)]:{p_more_than_5:.5f} ({p_more_than_5*100:.2f}%)")
    print(f"Theoretical Mean & Variance:     {rv.mean():.1f} & {rv.var():.1f}")

if __name__ == "__main__":
    main()
