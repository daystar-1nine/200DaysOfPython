"""
Production pipeline CLI orchestrator for Day 65 Statistical Analysis Engine.
"""

import time
from .config import DATA_PATH, OUTPUT_DIR, CHARTS_DIR, NUMERICAL_COLS
from .loader import load_dataset
from .validator import validate_and_clean_series
from .visualizations import generate_distribution_charts
from .report import generate_all_reports


def main():
    start_time = time.time()
    print("==================================================")
    print("   DAY 65: STATISTICAL ANALYSIS ENGINE PIPELINE   ")
    print("==================================================")
    
    print(f"[1/5] Loading dataset from: {DATA_PATH}")
    df = load_dataset(DATA_PATH)
    print(f"      Successfully loaded {len(df)} rows, {len(df.columns)} columns.")
    
    print("[2/5] Validating numerical variables and screening edge cases...")
    for col in NUMERICAL_COLS:
        if col in df.columns:
            cleaned_s, meta = validate_and_clean_series(df[col])
            print(f"      {col:<12}: {len(cleaned_s)} valid records (NaNs: {meta['nan_count']}, Infs: {meta['inf_count']})")
            
    print("[3/5] Generating publication-grade statistical diagnostic charts...")
    saved_charts = generate_distribution_charts(df, CHARTS_DIR)
    print(f"      Generated {len(saved_charts)} figures in {CHARTS_DIR}")
    
    print("[4/5] Executing statistical moments, dispersion & outlier engines...")
    reports = generate_all_reports(df, NUMERICAL_COLS, OUTPUT_DIR)
    print(f"      Saved: statistical_summary.csv")
    print(f"      Saved: percentile_report.csv")
    print(f"      Saved: outlier_report.csv")
    print(f"      Saved: zscore_report.csv")
    print(f"      Saved: statistical_report.txt")
    
    print("[5/5] Automated Insights Generated:")
    for i, ins in enumerate(reports["insights"], 1):
        print(f"      [{i}] {ins[:90]}...")
        
    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"Pipeline executed cleanly in {elapsed:.2f} seconds.")
    print("==================================================")


if __name__ == "__main__":
    main()
