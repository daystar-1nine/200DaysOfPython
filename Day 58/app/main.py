"""
Module: main.py
CLI Entry point for Messy Customer Dataset Cleaning Pipeline.
Coordinates data ingestion, cleaning pipeline, business validation, quality analysis, reporting, and file exports.
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
# Why it is used: Connects loader, cleaner, validator, analyzer, and report functions into main workflow.
# How it works: Imports load_data, clean_customer_data, validate_dataset, analyze_data_quality, etc.
from app.analyzer import analyze_data_quality
from app.cleaner import clean_customer_data
from app.loader import load_data
from app.report import export_clean_data_and_report, generate_quality_report
from app.validator import validate_dataset


def main() -> None:
    """
    Execute full Messy Customer Dataset Cleaning Pipeline workflow.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("================================================================================")
    print("           STARTING MESSY CUSTOMER DATASET CLEANING PIPELINE                    ")
    print("================================================================================")

    # What is used: Path resolution for raw CSV input and output target files.
    # Why it is used: Guarantees stable absolute path resolution regardless of shell CWD.
    # How it works: Resolves paths relative to Day 58 root.
    raw_csv_path = PROJECT_ROOT / "data" / "raw" / "messy_customers.csv"
    cleaned_csv_path = PROJECT_ROOT / "data" / "processed" / "clean_customers.csv"
    report_output_path = PROJECT_ROOT / "output" / "data_quality_report.txt"

    print(f"\n[1/5] Ingesting raw messy customer dataset from: {raw_csv_path}")
    raw_df = load_data(raw_csv_path)
    print(f"      Loaded {len(raw_df)} raw customer records across {len(raw_df.columns)} columns.")

    print("\n[2/5] Running automated data cleaning pipeline...")
    cleaned_df, cleaning_stats = clean_customer_data(raw_df)
    print(f"      Duplicate records dropped     : {cleaning_stats['duplicates_removed']}")
    print(f"      Invalid ages corrected (0-120): {cleaning_stats['invalid_ages_corrected']}")
    print(f"      Negative salaries corrected   : {cleaning_stats['invalid_salaries_corrected']}")
    print(f"      Invalid dates coerced to NaT  : {cleaning_stats['invalid_dates_corrected']}")
    print(f"      Total missing slots imputed   : {cleaning_stats['nulls_filled']}")
    print(f"      Final clean records remaining : {cleaning_stats['final_rows']}")

    print("\n[3/5] Validating cleaned dataset against business rules...")
    validation_results = validate_dataset(cleaned_df)
    status_str = "PASSED" if validation_results["is_valid"] else "FAILED"
    print(f"      Validation Audit Status: {status_str}")

    print("\n[4/5] Computing before/after data quality metrics...")
    quality_analysis = analyze_data_quality(raw_df, cleaned_df, cleaning_stats)

    print("\n[5/5] Generating Executive Data Quality Report & exporting cleaned data...")
    report_str = generate_quality_report(cleaning_stats, quality_analysis, validation_results)
    print("\n" + report_str)

    export_clean_data_and_report(cleaned_df, report_str, cleaned_csv_path, report_output_path)
    print(f"\n      Cleaned dataset exported to : {cleaned_csv_path}")
    print(f"      Executive report exported to: {report_output_path}")

    print("\n================================================================================")
    print("                 DATA CLEANING COMPLETED SUCCESSFULLY                           ")
    print("================================================================================")


if __name__ == "__main__":
    main()
