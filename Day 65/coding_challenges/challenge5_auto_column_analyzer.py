"""
Challenge 5 — Data Scientist Automatic Column Analyzer
Implements analyze_column(series) providing automated central tendency, dispersion, shape, and outlier audits.
"""

import pandas as pd
import numpy as np


def analyze_column(series: pd.Series) -> dict:
    clean_s = series.dropna()
    if clean_s.empty:
        return {"error": "Series contains no non-null values."}
    
    mean_val = float(clean_s.mean())
    median_val = float(clean_s.median())
    modes = clean_s.mode().tolist()
    std_val = float(clean_s.std(ddof=1)) if len(clean_s) > 1 else 0.0
    
    q1 = float(clean_s.quantile(0.25))
    q3 = float(clean_s.quantile(0.75))
    iqr_val = q3 - q1
    
    skew_val = float(clean_s.skew()) if len(clean_s) > 2 else 0.0
    kurt_val = float(clean_s.kurt()) if len(clean_s) > 3 else 0.0
    
    # Outlier Detection
    lower_fence = q1 - 1.5 * iqr_val
    upper_fence = q3 + 1.5 * iqr_val
    iqr_outliers = clean_s[(clean_s < lower_fence) | (clean_s > upper_fence)]
    
    # Z-Score Outliers
    if std_val > 0:
        z_scores = (clean_s - mean_val) / std_val
        z_outliers = clean_s[np.abs(z_scores) > 3.0]
    else:
        z_outliers = pd.Series([], dtype=float)
        
    # Shape Classification
    if abs(skew_val) < 0.5:
        skew_desc = "Approximately Symmetric"
    elif skew_val >= 0.5:
        skew_desc = "Moderately/Strongly Right-Skewed (Mean > Median)"
    else:
        skew_desc = "Moderately/Strongly Left-Skewed (Mean < Median)"
        
    if kurt_val > 1.0:
        kurt_desc = "Leptokurtic (Heavy Tails, High Outlier Risk)"
    elif kurt_val < -1.0:
        kurt_desc = "Platykurtic (Light Tails, Low Outlier Risk)"
    else:
        kurt_desc = "Mesokurtic (Normal-Like Tails)"

    return {
        "column_name": series.name if series.name else "Unnamed_Series",
        "sample_size": len(clean_s),
        "central_tendency": {
            "mean": round(mean_val, 4),
            "median": round(median_val, 4),
            "mode": modes[:3],
            "primary_center": "Median (due to skew)" if abs(skew_val) > 1.0 else "Mean"
        },
        "dispersion": {
            "range": round(float(clean_s.max() - clean_s.min()), 4),
            "variance": round(float(clean_s.var(ddof=1)), 4) if len(clean_s) > 1 else 0.0,
            "std_dev": round(std_val, 4),
            "iqr": round(iqr_val, 4),
            "coefficient_of_variation": round((std_val / mean_val) * 100.0, 2) if mean_val != 0 else np.nan
        },
        "distribution_shape": {
            "skewness": round(skew_val, 4),
            "skew_type": skew_desc,
            "kurtosis": round(kurt_val, 4),
            "kurtosis_type": kurt_desc
        },
        "outlier_audit": {
            "iqr_outlier_count": len(iqr_outliers),
            "iqr_outlier_pct": round((len(iqr_outliers) / len(clean_s)) * 100.0, 2),
            "zscore_outlier_count": len(z_outliers),
            "zscore_outlier_pct": round((len(z_outliers) / len(clean_s)) * 100.0, 2)
        }
    }


if __name__ == "__main__":
    import json
    test_sales = pd.Series([1200, 1500, 1800, 2000, 2100, 2200, 2500, 2800, 3000, 45000], name="Transaction_Revenue")
    analysis = analyze_column(test_sales)
    print("=== CHALLENGE 5: AUTOMATED DATA SCIENTIST COLUMN ANALYZER ===")
    print(json.dumps(analysis, indent=2))
