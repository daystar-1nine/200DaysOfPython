"""
Module: main.py
CLI Entry point for Sales Analytics Engine V1.
Coordinates data loading, cleaning pipeline, transformations, business analytics, reporting, and exports.
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
# Why it is used: Connects loader, cleaner, transformer, analyzer, and report functions into main workflow.
# How it works: Imports load_data, clean_data, transform_data, generate_analysis, generate_report, export_reports_and_summaries.
from app.analyzer import generate_analysis
from app.cleaner import clean_data
from app.loader import load_data
from app.report import export_reports_and_summaries, generate_report
from app.transformer import transform_data


def main() -> None:
    """
    Execute full Sales Analytics Engine workflow.
    """
    print("================================================================================")
    print("                  STARTING SALES ANALYTICS ENGINE V1                            ")
    print("================================================================================")

    # What is used: Path resolution for raw CSV input and output target files.
    # Why it is used: Guarantees stable absolute path resolution regardless of shell CWD.
    # How it works: Resolves paths relative to Day 57 root.
    raw_csv_path = PROJECT_ROOT / "data" / "raw" / "sales.csv"
    cleaned_csv_path = PROJECT_ROOT / "data" / "processed" / "cleaned_sales.csv"
    report_output_path = PROJECT_ROOT / "output" / "sales_report.txt"
    pivot_csv_path = PROJECT_ROOT / "output" / "regional_category_summary.csv"

    print(f"\n[1/5] Ingesting raw sales transaction dataset from: {raw_csv_path}")
    raw_df = load_data(raw_csv_path)
    print(f"      Loaded {len(raw_df)} raw sales records across {len(raw_df.columns)} columns.")

    print("\n[2/5] Cleaning dataset (removing duplicates, imputing missing numeric values, validating ranges)...")
    cleaned_df, cleaning_stats = clean_data(raw_df)
    print(f"      Duplicates removed: {cleaning_stats['duplicates_removed']}")
    print(f"      Missing numeric values imputed: {cleaning_stats['nulls_filled']}")
    print(f"      Invalid records dropped: {cleaning_stats['invalid_rows_dropped']}")
    print(f"      Clean records remaining: {cleaning_stats['final_rows']}")

    print("\n[3/5] Transforming data (calculating Revenue, parsing datetime, extracting Month periods, binning discounts)...")
    transformed_df = transform_data(cleaned_df)
    print(f"      Successfully calculated net revenue for {len(transformed_df)} transactions.")

    print("\n[4/5] Computing business analytics, regional/category breakdowns, and pivot tables...")
    results, pivot_df = generate_analysis(transformed_df)
    overall = results["overall"]
    print(f"      Total Grand Revenue: ${overall['total_revenue']:,.2f}")
    print(f"      Average Order Value: ${overall['average_order_value']:,.2f}")

    print("\n[5/5] Generating executive ASCII report & exporting results...")
    report_str = generate_report(results, cleaning_stats)
    print("\n" + report_str)

    export_reports_and_summaries(
        transformed_df, pivot_df, report_str,
        cleaned_csv_path, report_output_path, pivot_csv_path
    )
    print(f"\n      Cleaned dataset exported to : {cleaned_csv_path}")
    print(f"      Executive report exported to: {report_output_path}")
    print(f"      Pivot summary exported to   : {pivot_csv_path}")

    print("\n================================================================================")
    print("                    SALES ANALYSIS COMPLETED SUCCESSFULLY                       ")
    print("================================================================================")


if __name__ == "__main__":
    main()
