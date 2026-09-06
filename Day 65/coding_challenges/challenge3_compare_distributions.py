"""
Challenge 3 — Comparative Distribution Analysis
Implements compare_distributions(series1, series2) across 5 core metrics with automated delta analysis.
"""

import pandas as pd
import numpy as np


def compare_distributions(s1: pd.Series, s2: pd.Series, name1: str = "Series 1", name2: str = "Series 2") -> pd.DataFrame:
    c1 = s1.dropna()
    c2 = s2.dropna()
    
    metrics = ["Mean", "Median", "Std", "IQR", "Skewness"]
    
    iqr1 = c1.quantile(0.75) - c1.quantile(0.25)
    iqr2 = c2.quantile(0.75) - c2.quantile(0.25)
    
    val1 = [c1.mean(), c1.median(), c1.std(), iqr1, c1.skew()]
    val2 = [c2.mean(), c2.median(), c2.std(), iqr2, c2.skew()]
    
    diff = [v2 - v1 for v1, v2 in zip(val1, val2)]
    pct_change = [(d / abs(v1)) * 100.0 if v1 != 0 else np.nan for v1, d in zip(val1, diff)]
    
    df_comp = pd.DataFrame({
        "Metric": metrics,
        name1: [round(v, 4) for v in val1],
        name2: [round(v, 4) for v in val2],
        "Difference": [round(d, 4) for d in diff],
        "Pct_Change (%)": [round(p, 2) for p in pct_change]
    })
    return df_comp


if __name__ == "__main__":
    np.random.seed(42)
    s_retail = pd.Series(np.random.normal(loc=100, scale=20, size=200), name="Retail")
    s_enterprise = pd.Series(np.random.lognormal(mean=5.0, sigma=0.6, size=200), name="Enterprise")
    
    comparison = compare_distributions(s_retail, s_enterprise, "Retail_Sales", "Enterprise_Sales")
    print("=== CHALLENGE 3: DISTRIBUTION COMPARISON ===")
    print(comparison.to_string(index=False))
