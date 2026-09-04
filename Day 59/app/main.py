"""
Module: main.py
CLI Entry point for E-Commerce Exploratory Data Analysis (EDA) Engine.
Coordinates data ingestion, clean-up, financial metric transformation, descriptive statistics, group analytics, time-series, correlation, outliers, reporting, and file exports.
"""

# What is used: Import sys module and pathlib Path class.
# Why it is used: Resolves project root path and configures system import paths dynamically.
# How it works: Prepends parent directory to sys.path for clean import resolution.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# What is used: Import application pipeline modules.
# Why it is used: Connects loader, cleaner, transformer, descriptive, group, time, correlation, outlier, and report functions.
# How it works: Imports pipeline function definitions into execution scope.
from app.cleaner import clean_sales_data
from app.correlation import compute_correlation_matrix
from app.descriptive import compute_descriptive_stats
from app.group_analysis import (
    analyze_category_performance,
    analyze_customer_performance,
    analyze_product_performance,
    analyze_regional_performance,
)
from app.loader import load_raw_dataset
from app.outliers import detect_all_outliers
from app.report import export_eda_artifacts, generate_eda_report
from app.time_analysis import analyze_monthly_trends
from app.transformer import compute_derived_metrics


def main() -> None:
    """
    Execute full 12-Phase E-Commerce Exploratory Data Analysis (EDA) Pipeline workflow.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("================================================================================")
    print("           STARTING E-COMMERCE EXPLORATORY DATA ANALYSIS ENGINE                 ")
    print("================================================================================")

    # Path resolution for raw CSV input and output target files
    raw_csv_path = PROJECT_ROOT / "data" / "raw" / "ecommerce_sales.csv"

    print(f"\n[1/12] Ingesting raw transactional dataset from: {raw_csv_path}")
    raw_df = load_raw_dataset(raw_csv_path)
    print(f"       Loaded {len(raw_df)} raw sales records across {len(raw_df.columns)} columns.")

    print("\n[2/12] Running automated data cleaning & deduplication pipeline...")
    cleaned_df, cleaning_stats = clean_sales_data(raw_df)
    print(f"       Duplicate records dropped : {cleaning_stats['duplicates_removed']}")
    print(f"       Total missing imputed     : {cleaning_stats['nulls_imputed']}")
    print(f"       Final clean records       : {cleaning_stats['final_rows']}")

    print("\n[3/12] Computing derived financial metrics (Revenue, Cost, Profit) & date features...")
    enriched_df = compute_derived_metrics(cleaned_df)

    print("\n[4/12] Calculating descriptive statistics, quantiles & IQR metrics...")
    desc_stats = compute_descriptive_stats(enriched_df)

    print("\n[5/12] Analyzing regional sales & profit performance...")
    regional_df = analyze_regional_performance(enriched_df)

    print("\n[6/12] Analyzing product category revenue & discount distribution...")
    category_df = analyze_category_performance(enriched_df)

    print("\n[7/12] Analyzing product performance & intra-category rankings...")
    top_rev, top_qty, top_prof = analyze_product_performance(enriched_df)

    print("\n[8/12] Analyzing customer lifetime spend, AOV & spend ranks...")
    customer_df = analyze_customer_performance(enriched_df)

    print("\n[9/12] Computing time-series monthly trends, MoM growth % & 3-month rolling averages...")
    monthly_df = analyze_monthly_trends(enriched_df)

    print("\n[10/12] Computing Pearson correlation matrix across numerical metrics...")
    corr_df = compute_correlation_matrix(enriched_df)

    print("\n[11/12] Auditing statistical outliers using Interquartile Range (IQR) bounds...")
    outliers_dict = detect_all_outliers(enriched_df)

    print("\n[12/12] Generating Executive EDA Report with 10+ Business Insights & exporting artifacts...")
    report_str = generate_eda_report(
        cleaning_stats, desc_stats, regional_df, category_df,
        top_rev, customer_df, monthly_df, corr_df, outliers_dict
    )
    print("\n" + report_str)

    export_eda_artifacts(
        enriched_df, report_str, regional_df, category_df,
        customer_df, top_rev, monthly_df, PROJECT_ROOT
    )

    print(f"\n       Cleaned dataset exported to: {PROJECT_ROOT / 'data' / 'processed' / 'cleaned_sales.csv'}")
    print(f"       Executive report exported to : {PROJECT_ROOT / 'output' / 'eda_report.txt'}")
    print(f"       CSV summaries exported to   : {PROJECT_ROOT / 'output'}")

    print("\n================================================================================")
    print("                 EXPLORATORY DATA ANALYSIS COMPLETED SUCCESSFULLY               ")
    print("================================================================================")


if __name__ == "__main__":
    main()
