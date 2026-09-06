"""
Day 67 Challenge 2: Normal Percent Point Function (PPF / Quantile) Analysis
Computes key percentile thresholds for IQ / test scores ~ N(mu=100, sigma=15).
"""
import pandas as pd
from scipy.stats import norm

def calculate_normal_quantiles(mu: float = 100.0, sigma: float = 15.0) -> pd.DataFrame:
    percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    records = []
    
    rv = norm(loc=mu, scale=sigma)
    for q in percentiles:
        val = float(rv.ppf(q))
        z_score = (val - mu) / sigma
        records.append({
            "Percentile (%)": f"P{int(q*100):02d}",
            "Cumulative Prob": q,
            "Threshold Value": round(val, 2),
            "Z-Score": round(z_score, 2),
            "Interpretation": "Median / Center" if q == 0.5 else ("Below Average" if q < 0.5 else "Above Average")
        })
        
    return pd.DataFrame(records)

def main():
    print("=" * 65)
    print("  CHALLENGE 2: PPF QUANTILE MAPPING (mu = 100, sigma = 15)")
    print("=" * 65)
    df_quantiles = calculate_normal_quantiles(100.0, 15.0)
    print(df_quantiles.to_string(index=False))
    print("\nKey Observation:")
    print("- Due to symmetry, P50 = 100.0 (Z = 0.0).")
    print("- Interquartile Range (IQR = P75 - P25) = 110.12 - 89.88 = 20.24 = 1.349 * sigma.")
    print("- P99 cutoff is 134.90 (Z = +2.33), encompassing 99% of the population.")

if __name__ == "__main__":
    main()
