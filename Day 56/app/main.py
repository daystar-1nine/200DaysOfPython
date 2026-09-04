"""
Module: main.py
CLI Entry point for Student Performance Analyzer V2.
Coordinates data loading, cleaning pipeline, analytical processing, reporting, and file exports.
"""

# What is used: Import sys module and pathlib Path class.
# Why it is used: Resolves project root directory dynamically and manages system imports.
# How it works: Adds parent project path to sys.path if invoked directly.
import sys
from pathlib import Path

# Add project root directory to path for clean import resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# What is used: Import project pipeline functions from app package.
# Why it is used: Connects pipeline modules into a cohesive CLI workflow.
# How it works: Imports load_student_csv, clean_student_data, analyze_student_performance, etc.
from app.analyzer import analyze_student_performance
from app.cleaner import clean_student_data
from app.loader import load_student_csv
from app.report import export_analysis_results, generate_ascii_report


def main() -> None:
    """
    Execute full Student Performance Analyzer V2 workflow.
    """
    print("================================================================================")
    print("                STARTING STUDENT PERFORMANCE ANALYZER V2                        ")
    print("================================================================================")

    # What is used: Path resolution for input raw dataset and output files.
    # Why it is used: Ensures relative path stability regardless of execution working directory.
    # How it works: Constructs absolute paths relative to Day 56 root folder.
    raw_csv_path = PROJECT_ROOT / "data" / "raw" / "students.csv"
    processed_csv_path = PROJECT_ROOT / "data" / "processed" / "student_analysis.csv"
    report_output_path = PROJECT_ROOT / "output" / "report.txt"

    print(f"\n[1/5] Loading raw student dataset from: {raw_csv_path}")
    raw_df = load_student_csv(raw_csv_path)
    print(f"      Loaded {len(raw_df)} raw records across {len(raw_df.columns)} columns.")

    print("\n[2/5] Cleaning dataset (removing duplicates, imputing missing marks, validating ranges)...")
    cleaned_df, cleaning_stats = clean_student_data(raw_df)
    print(f"      Duplicates removed: {cleaning_stats['duplicates_removed']}")
    print(f"      Missing marks imputed: {cleaning_stats['nulls_filled']}")
    print(f"      Invalid scores dropped: {cleaning_stats['invalid_scores_dropped']}")
    print(f"      Clean records remaining: {cleaning_stats['final_rows']}")

    print("\n[3/5] Performing academic performance analytics & grade assignments...")
    analysis_df, metrics = analyze_student_performance(cleaned_df)
    print(f"      Overall Class Average: {metrics['overall_average']} / 100")
    print(f"      Passing Rate: {metrics['pass_rate']}% ({metrics['pass_count']}/{metrics['total_students']})")

    print("\n[4/5] Generating executive ASCII report...")
    report_str = generate_ascii_report(metrics, cleaning_stats)
    print("\n" + report_str)

    print("\n[5/5] Exporting processed results and report...")
    export_analysis_results(analysis_df, report_str, processed_csv_path, report_output_path)
    print(f"      Processed dataset exported to: {processed_csv_path}")
    print(f"      Executive report exported to : {report_output_path}")

    print("\n================================================================================")
    print("                       ANALYSIS COMPLETED SUCCESSFULLY                          ")
    print("================================================================================")


if __name__ == "__main__":
    main()
