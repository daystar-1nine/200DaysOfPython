"""
Module: report.py
Generates ASCII Data Quality Reports and exports cleaned customer datasets.
"""

# What is used: Import os module and pathlib Path class.
# Why it is used: Manages directory creation and file writing.
# How it works: Ensures output directories exist before saving files.
import os
from pathlib import Path

# What is used: Import pandas library.
# Why it is used: Handles CSV exporting via df.to_csv().
# How it works: Serializes DataFrame objects to disk.
import pandas as pd


def generate_quality_report(stats: dict, analysis: dict, validation: dict) -> str:
    """
    Format data quality audit metrics, before/after comparisons, and validation flags into an ASCII report.

    Args:
        stats: Cleaner audit statistics dictionary.
        analysis: Data quality analytical metrics dictionary.
        validation: Validation rule audit dictionary.

    Returns:
        str: Formatted ASCII text report string.
    """
    missing_info = analysis.get("missing", {})
    dup_info = analysis.get("duplicates", {})
    desc_info = analysis.get("descriptive", {})
    demo_info = analysis.get("demographics", {})

    report_lines = [
        "================================================================================",
        "                       DATA QUALITY & CLEANING REPORT                           ",
        "================================================================================",
        "",
        "--- 1. RECORD COUNT & DEDUPLICATION AUDIT ---",
        f"  * Initial Raw Records   : {stats.get('initial_rows', 0)}",
        f"  * Duplicate IDs Removed : {stats.get('duplicates_removed', 0)}",
        f"  * Final Clean Records   : {stats.get('final_rows', 0)}",
        "",
        "--- 2. DOMAIN RANGE & INVALID DATA CORRECTIONS ---",
        f"  * Out-of-Bounds Ages Corrected (0-120) : {stats.get('invalid_ages_corrected', 0)}",
        f"  * Negative Salaries Corrected (< 0)    : {stats.get('invalid_salaries_corrected', 0)}",
        f"  * Unparseable Dates Coerced to NaT     : {stats.get('invalid_dates_corrected', 0)}",
        f"  * Total Null Slots Imputed             : {stats.get('nulls_filled', 0)}",
        "",
        "--- 3. MISSING VALUE AUDIT (BEFORE VS AFTER CLEANING) ---",
        f"    {'Column':<16} {'Raw NaNs':<10} {'Raw NaNs (%)':<14} {'Clean NaNs':<12} {'Clean NaNs (%)':<14}",
        "    " + "-" * 66
    ]

    raw_c = missing_info.get("raw_count", {})
    raw_p = missing_info.get("raw_pct", {})
    clean_c = missing_info.get("clean_count", {})
    clean_p = missing_info.get("clean_pct", {})

    for col in raw_c.keys():
        r_cnt = raw_c.get(col, 0)
        r_pct = raw_p.get(col, 0.0)
        c_cnt = clean_c.get(col, 0)
        c_pct = clean_p.get(col, 0.0)
        report_lines.append(
            f"    {col:<16} {r_cnt:<10} {r_pct:<13.2f}% {c_cnt:<12} {c_pct:<13.2f}%"
        )

    report_lines.extend([
        "",
        "--- 4. DESCRIPTIVE STATISTICS (POST-CLEANING) ---"
    ])

    for col, s in desc_info.items():
        report_lines.append(
            f"  * {col.title():<8} -> Mean: {s.get('mean', 0.0):<10} Median: {s.get('median', 0.0):<10} "
            f"Min: {s.get('min', 0.0):<8} Max: {s.get('max', 0.0):<8} Std: {s.get('std', 0.0)}"
        )

    report_lines.extend([
        "",
        "--- 5. DEMOGRAPHIC & CATEGORICAL DISTRIBUTIONS ---",
        f"  * Gender Distribution    : {demo_info.get('gender', {})}",
        f"  * Department Breakdown   : {demo_info.get('department', {})}",
        f"  * Top Cities Breakdown   : {demo_info.get('city', {})}",
        "",
        "--- 6. BUSINESS DATA RULE VALIDATION STATUS ---",
        f"  * Overall Dataset Status : {'PASSED (Clean & Analysis-Ready)' if validation.get('is_valid', False) else 'FAILED'}"
    ])

    for r_name, r_data in validation.get("rules", {}).items():
        status_str = "PASS" if r_data.get("passed", False) else "FAIL"
        report_lines.append(f"  * Rule [{r_name:<24}]: {status_str}")

    report_lines.append("================================================================================")
    return "\n".join(report_lines)


def export_clean_data_and_report(
    clean_df: pd.DataFrame,
    report_str: str,
    clean_csv_path: str | Path,
    report_txt_path: str | Path
) -> None:
    """
    Export processed clean customer CSV and formatted data quality text report.

    Args:
        clean_df: Processed clean customer DataFrame.
        report_str: Formatted ASCII report string.
        clean_csv_path: Target path for clean CSV export.
        report_txt_path: Target path for report text export.
    """
    # What is used: Path.mkdir(parents=True, exist_ok=True).
    # Why it is used: Ensures parent destination folders exist before writing.
    # How it works: Recursively creates missing directory tree.
    c_path = Path(clean_csv_path)
    r_path = Path(report_txt_path)

    c_path.parent.mkdir(parents=True, exist_ok=True)
    r_path.parent.mkdir(parents=True, exist_ok=True)

    # What is used: df.to_csv(index=False).
    # Why it is used: Saves cleaned customer DataFrame to CSV.
    # How it works: Writes dataset rows to disk.
    clean_df.to_csv(c_path, index=False)

    # What is used: File open() with utf-8 encoding.
    # Why it is used: Saves formatted ASCII report to text file safely.
    # How it works: Writes report string to file.
    with open(r_path, "w", encoding="utf-8") as f:
        f.write(report_str)
