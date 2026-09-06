"""
Day 68 Coding Challenge 5: E-Commerce Average Order Value (AOV) Estimation
Analyzes customer transactions in data/sample_data.csv across sample sizes,
balancing statistical precision (Margin of Error) against sampling cost.
"""
from pathlib import Path
import numpy as np
import pandas as pd

def analyze_aov_sampling_budget(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    population_values = df["order_value"].values
    
    N = len(population_values)
    true_mean = float(np.mean(population_values))
    true_std = float(np.std(population_values))
    
    candidate_sample_sizes = [50, 100, 250, 500, 1000, 2000]
    cost_per_audit = 2.0  # $2 audit verification cost per sampled record
    
    records = []
    rng = np.random.default_rng(42)
    
    for n in candidate_sample_sizes:
        sample = rng.choice(population_values, size=n, replace=False)
        sample_mean = float(np.mean(sample))
        sample_std = float(np.std(sample, ddof=1))
        
        # Theoretical SE & 95% Margin of Error (1.96 * SE)
        se = true_std / np.sqrt(n)
        margin_of_error = 1.96 * se
        total_cost = n * cost_per_audit
        
        records.append({
            "Sample Size (n)": n,
            "Total Cost ($)": f"${total_cost:,.0f}",
            "Sample AOV ($)": round(sample_mean, 2),
            "True AOV ($)": round(true_mean, 2),
            "Estimation Error ($)": round(abs(sample_mean - true_mean), 2),
            "Standard Error ($)": round(se, 2),
            "95% Margin of Error ($)": f"+/- ${margin_of_error:.2f}",
            "Relative Precision (%)": f"{(margin_of_error / true_mean) * 100:.1f}%"
        })
        
    return pd.DataFrame(records)

def main():
    print("=" * 80)
    print("  CHALLENGE 5: E-COMMERCE AOV SAMPLING BUDGET & PRECISION ANALYSIS")
    print("=" * 80)
    data_path = Path(__file__).resolve().parent.parent / "data" / "sample_data.csv"
    df_budget = analyze_aov_sampling_budget(data_path)
    print(df_budget.to_string(index=False))
    print("-" * 80)
    print("Recommendation:")
    print("- n = 500 achieves a 95% Margin of Error of ~ $3.40 (4.5% of AOV) for $1,000.")
    print("- Diminishing returns: Quadrupling budget to $4,000 (n=2000) only halves error to $1.70.")

if __name__ == "__main__":
    main()
