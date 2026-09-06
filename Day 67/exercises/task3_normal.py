"""
Day 67 Task 3: Normal Distribution Exam Score Probabilities
Evaluates normal exam scores ~ N(mu=70, sigma=10),
calculating tail and interval probabilities.
"""
from scipy.stats import norm

def main():
    mu = 70
    sigma = 10
    rv = norm(loc=mu, scale=sigma)
    
    p_less_60 = float(rv.cdf(60))
    p_greater_80 = float(rv.sf(80))  # 1 - cdf(80)
    p_between_60_80 = float(rv.cdf(80) - rv.cdf(60))
    
    print("=" * 60)
    print("  TASK 3: NORMAL DISTRIBUTION (mu = 70, sigma = 10)")
    print("=" * 60)
    print(f"P(Score < 60) [Below 1 Std Dev]:     {p_less_60:.5f} ({p_less_60*100:.2f}%)")
    print(f"P(Score > 80) [Above 1 Std Dev]:     {p_greater_80:.5f} ({p_greater_80*100:.2f}%)")
    print(f"P(60 < Score < 80) [Within 1 Std]:   {p_between_60_80:.5f} ({p_between_60_80*100:.2f}%)")

if __name__ == "__main__":
    main()
