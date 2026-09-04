"""
Module: main.py
CLI Entry point for Day 60 Business Intelligence Analytics Engine.
Orchestrates data ingestion, cleaning, validation, transformation, multidimensional analytics, statistical modeling, automated insight synthesis, and report artifact exports.
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
# Why it is used: Connects loader, cleaner, validator, transformer, analysis subpackage, insights, and report modules.
# How it works: Brings all engine components into execution scope.
from app.analysis.category import analyze_categories
from app.analysis.customer import analyze_customers
from app.analysis.outliers import detect_outliers_iqr
from app.analysis.overview import compute_overview_kpis
from app.analysis.product import analyze_products
from app.analysis.regional import analyze_regions
from app.analysis.statistics import compute_correlations, compute_numerical_statistics
from app.analysis.time_series import analyze_daily_series, analyze_monthly_series
from app.cleaner import clean_sales_records
from app.config import OUTPUT_DIR, PROCESSED_DATA_PATH, RAW_DATA_PATH
from app.insights import generate_business_insights
from app.loader import load_dataset
from app.reports import export_all_artifacts, generate_data_quality_report, generate_executive_summary
from app.transformer import transform_sales_data
from app.validator import validate_sales_data


def main() -> None:
    """
    Execute end-to-end Day 60 Business Intelligence Analytics Pipeline.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("================================================================================")
    print("      STARTING DAY 60 BUSINESS INTELLIGENCE ANALYTICS ENGINE (CAPSTONE)         ")
    print("================================================================================")

    # 1. Data Ingestion
    print(f"\n[1/10] Ingesting raw transactional sales dataset from: {RAW_DATA_PATH}")
    raw_df = load_dataset(RAW_DATA_PATH)
    print(f"       Loaded {len(raw_df)} raw sales records across {len(raw_df.columns)} columns.")

    # 2. Data Cleaning
    print("\n[2/10] Running automated cleaning & deduplication pipeline...")
    cleaned_df, clean_audit = clean_sales_records(raw_df)
    print(f"       Duplicates dropped     : {clean_audit['duplicates_removed']}")
    print(f"       Missing slots imputed  : {clean_audit['nulls_filled']}")
    print(f"       Dates coerced          : {clean_audit['dates_coerced']}")
    print(f"       Final clean rows       : {clean_audit['final_rows']}")

    # 3. Data Validation
    print("\n[3/10] Auditing dataset against business domain rules...")
    val_results = validate_sales_data(cleaned_df)
    val_status = "PASSED (Analysis Ready)" if val_results["is_valid"] else "FAILED"
    print(f"       Validation Status      : {val_status}")

    # 4. Feature Creation / Transformation
    print("\n[4/10] Computing financial KPIs (Revenue, Cost, Profit, Margins) & date features...")
    enriched_df = transform_sales_data(cleaned_df)

    # 5. Multidimensional Business Analysis
    print("\n[5/10] Executing corporate overview, regional, and category analysis...")
    overview_kpis = compute_overview_kpis(enriched_df)
    regional_df = analyze_regions(enriched_df)
    category_df = analyze_categories(enriched_df)

    print("\n[6/10] Executing customer lifetime value and product catalog rankings...")
    customer_df = analyze_customers(enriched_df)
    product_df, top_rev, top_qty, top_prof = analyze_products(enriched_df)

    # 6. Time-Series Trends
    print("\n[7/10] Computing monthly trends, MoM growth rates & 3-month moving averages...")
    monthly_df = analyze_monthly_series(enriched_df)
    daily_df = analyze_daily_series(enriched_df)

    # 7. Statistical Modeling & Correlations
    print("\n[8/10] Computing descriptive statistics, quantiles, and correlation matrices...")
    stats_dict = compute_numerical_statistics(enriched_df)
    corr_df, cov_df = compute_correlations(enriched_df)

    # 8. Outlier Detection
    print("\n[9/10] Performing IQR outlier audits across continuous transaction features...")
    outliers_dict = detect_outliers_iqr(enriched_df)

    # 9. Automated Dynamic Business Insights
    print("\n[10/10] Synthesizing automated dynamic business insights & exporting artifacts...")
    insights = generate_business_insights(
        overview_kpis, regional_df, category_df, customer_df,
        top_rev, monthly_df, corr_df, outliers_dict
    )

    # Format Reports
    exec_summary_str = generate_executive_summary(
        overview_kpis, regional_df, category_df, top_rev,
        customer_df, monthly_df, stats_dict, insights
    )
    data_quality_str = generate_data_quality_report(clean_audit, val_results, outliers_dict)

    print("\n" + exec_summary_str)

    # Export Artifacts
    export_all_artifacts(
        enriched_df, exec_summary_str, data_quality_str,
        regional_df, category_df, product_df, customer_df,
        monthly_df, PROCESSED_DATA_PATH, OUTPUT_DIR
    )

    print(f"\n       Cleaned dataset exported to: {PROCESSED_DATA_PATH}")
    print(f"       Executive summary exported to: {OUTPUT_DIR / 'executive_summary.txt'}")
    print(f"       Quality report exported to   : {OUTPUT_DIR / 'data_quality_report.txt'}")
    print(f"       CSV analytical summaries     : {OUTPUT_DIR}")

    print("\n================================================================================")
    print("           DAY 60 MILESTONE ANALYTICS COMPLETED SUCCESSFULLY                    ")
    print("================================================================================")


if __name__ == "__main__":
    main()
