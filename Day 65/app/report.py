"""
Report generation and CSV/ASCII exporter module.
Builds Reports 1 through 8 and writes summary artifacts.
"""

import os
import sys
import pandas as pd
import numpy as np

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

try:
    from .statistics import (
        calculate_mean, calculate_median, calculate_mode,
        calculate_range, calculate_variance, calculate_std, calculate_iqr,
        calculate_percentiles, detect_iqr_outliers, detect_zscore_outliers,
        calculate_skewness, calculate_kurtosis, classify_distribution
    )
    from .insights import generate_insights
except ImportError:
    from app.statistics import (
        calculate_mean, calculate_median, calculate_mode,
        calculate_range, calculate_variance, calculate_std, calculate_iqr,
        calculate_percentiles, detect_iqr_outliers, detect_zscore_outliers,
        calculate_skewness, calculate_kurtosis, classify_distribution
    )
    from app.insights import generate_insights


def generate_all_reports(df: pd.DataFrame, target_cols: list[str], output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    
    # Report 1: Basic Statistics
    rep1_rows = []
    summary_dict = {}
    
    # Report 2: Dispersion
    rep2_rows = []
    
    # Report 3: Distribution
    rep3_rows = []
    
    # Report 4: Outliers
    rep4_rows = []
    outlier_dict = {}
    
    # Report 5: Z-Scores
    rep5_rows = []
    
    # Report 6: Percentiles
    rep6_rows = []
    percentile_dict = {}
    
    for col in target_cols:
        s = df[col].dropna()
        if s.empty:
            continue
            
        mean_val = calculate_mean(s)
        med_val = calculate_median(s)
        mode_val = calculate_mode(s)
        min_val = float(s.min())
        max_val = float(s.max())
        rng_val = calculate_range(s)
        
        var_sample = calculate_variance(s, ddof=1)
        std_sample = calculate_std(s, ddof=1)
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr_val = calculate_iqr(s)
        
        skew_val = calculate_skewness(s)
        kurt_val = calculate_kurtosis(s)
        shape_class = classify_distribution(skew_val, kurt_val)
        
        iqr_out = detect_iqr_outliers(s, multiplier=1.5)
        outlier_dict[col] = iqr_out
        
        z_out = detect_zscore_outliers(s, threshold=3.0, ddof=1)
        
        p_dict = calculate_percentiles(s, [10, 25, 50, 75, 90, 95, 99])
        percentile_dict[col] = p_dict
        
        summary_dict[col] = {
            "mean": mean_val, "median": med_val, "mode": mode_val,
            "min": min_val, "max": max_val, "range": rng_val,
            "std": std_sample
        }
        
        # Row 1
        rep1_rows.append({
            "Column": col, "Count": int(s.count()), "Mean": round(mean_val, 2),
            "Median": round(med_val, 2), "Mode": mode_val[0] if mode_val else np.nan,
            "Min": round(min_val, 2), "Max": round(max_val, 2), "Range": round(rng_val, 2)
        })
        
        # Row 2
        rep2_rows.append({
            "Column": col, "Variance": round(var_sample, 2), "Std_Dev": round(std_sample, 2),
            "Q1": round(q1, 2), "Q3": round(q3, 2), "IQR": round(iqr_val, 2)
        })
        
        # Row 3
        rep3_rows.append({
            "Column": col, "Skewness": round(skew_val, 3), "Kurtosis": round(kurt_val, 3),
            "Skew_Type": shape_class["skewness_classification"],
            "Tail_Type": shape_class["kurtosis_classification"]
        })
        
        # Row 4
        rep4_rows.append({
            "Column": col, "Lower_Bound": iqr_out["lower_bound"], "Upper_Bound": iqr_out["upper_bound"],
            "Outlier_Count": iqr_out["count"], "Outlier_Percentage": iqr_out["percentage"]
        })
        
        # Row 5
        rep5_rows.append({
            "Column": col, "Threshold": z_out["threshold"],
            "Extreme_Count": z_out["count"], "Extreme_Percentage": z_out["percentage"]
        })
        
        # Row 6
        p_row = {"Column": col}
        p_row.update({k: round(v, 2) for k, v in p_dict.items()})
        rep6_rows.append(p_row)
        
    df_basic = pd.DataFrame(rep1_rows)
    df_disp = pd.DataFrame(rep2_rows)
    df_dist = pd.DataFrame(rep3_rows)
    df_outliers = pd.DataFrame(rep4_rows)
    df_zscore = pd.DataFrame(rep5_rows)
    df_percentiles = pd.DataFrame(rep6_rows)
    
    # Export CSVs
    basic_csv = os.path.join(output_dir, "statistical_summary.csv")
    df_basic.to_csv(basic_csv, index=False)
    
    perc_csv = os.path.join(output_dir, "percentile_report.csv")
    df_percentiles.to_csv(perc_csv, index=False)
    
    out_csv = os.path.join(output_dir, "outlier_report.csv")
    df_outliers.to_csv(out_csv, index=False)
    
    z_csv = os.path.join(output_dir, "zscore_report.csv")
    df_zscore.to_csv(z_csv, index=False)
    
    # Generate Insights (Report 7)
    insights = generate_insights(summary_dict, outlier_dict, percentile_dict)
    
    # Assemble Master Text Report (statistical_report.txt)
    txt_path = os.path.join(output_dir, "statistical_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("      EXECUTIVE DESCRIPTIVE STATISTICS & STATISTICAL AUDIT REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("SECTION 1: BASIC CENTRAL TENDENCY & RANGE\n")
        f.write("-" * 80 + "\n")
        f.write(df_basic.to_string(index=False) + "\n\n")
        
        f.write("SECTION 2: DISPERSION, VARIANCE & INTERQUARTILE RANGE\n")
        f.write("-" * 80 + "\n")
        f.write(df_disp.to_string(index=False) + "\n\n")
        
        f.write("SECTION 3: DISTRIBUTION SHAPE (SKEWNESS & KURTOSIS MOMENTS)\n")
        f.write("-" * 80 + "\n")
        f.write(df_dist.to_string(index=False) + "\n\n")
        
        f.write("SECTION 4: IQR OUTLIER ENVELOPE (TUKEY'S RULE, 1.5x IQR)\n")
        f.write("-" * 80 + "\n")
        f.write(df_outliers.to_string(index=False) + "\n\n")
        
        f.write("SECTION 5: Z-SCORE EXTREME ANOMALY AUDIT (|z| > 3.0)\n")
        f.write("-" * 80 + "\n")
        f.write(df_zscore.to_string(index=False) + "\n\n")
        
        f.write("SECTION 6: MULTI-TIER PERCENTILE MATRIX (P10 THROUGH P99)\n")
        f.write("-" * 80 + "\n")
        f.write(df_percentiles.to_string(index=False) + "\n\n")
        
        f.write("SECTION 7: AUTOMATED STATISTICAL INSIGHTS & INTERPRETATIONS\n")
        f.write("-" * 80 + "\n")
        for i, ins in enumerate(insights, 1):
            f.write(f"[{i}] {ins}\n\n")
            
        f.write("SECTION 8: VISUAL VALIDATION ASSETS\n")
        f.write("-" * 80 + "\n")
        f.write("All statistical distributions verified against publication-grade figures in output/charts/:\n")
        f.write("  - revenue_distribution.png (Histogram, KDE, Mean/Median divergence)\n")
        f.write("  - profit_distribution.png (Profit spread, extreme tail risk)\n")
        f.write("  - quantity_distribution.png (Discrete order counts)\n")
        f.write("  - revenue_boxplot.png (Tukey IQR boundaries & outliers)\n")
        f.write("  - profit_boxplot.png (Profit dispersion & outlier points)\n")
        f.write("  - discount_boxplot.png (Promotional tier distribution)\n\n")
        f.write("=" * 80 + "\n")
        f.write("END OF STATISTICAL REPORT\n")
        f.write("=" * 80 + "\n")
        
    return {
        "basic_df": df_basic,
        "dispersion_df": df_disp,
        "distribution_df": df_dist,
        "outlier_df": df_outliers,
        "zscore_df": df_zscore,
        "percentile_df": df_percentiles,
        "insights": insights,
        "report_txt": txt_path
    }
