"""
===============================================================================
DAY 53 — MAIN APPLICATION ENTRY POINT
===============================================================================
This module executes the Sales Data Processing System pipeline and provides an
interactive CLI menu for pipeline execution and analysis inspection.
===============================================================================
"""

import sys
from pathlib import Path
from typing import List
from app.models import Sale
from app.csv_loader import load_raw_csv
from app.transformer import transform_and_deduplicate
from app.analyzer import (
    total_revenue,
    average_order_value,
    highest_value_order,
    best_selling_product,
    category_revenue,
    top_products_by_revenue,
)
from app.reporter import export_cleaned_csv, generate_sales_report


def run_pipeline(raw_csv_path: Path, processed_csv_path: Path, report_path: Path) -> List[Sale]:
    """Execute the end-to-end data processing pipeline."""
    # What is used: Modular functional pipeline orchestration.
    # Why it is used: Ingests raw data, cleans, validates, deduplicates, analyzes, and exports reports.
    # How it works: Calls csv_loader -> transformer -> reporter -> returns sales list.
    print("\n[INFO] Starting Sales Data Processing Pipeline...")
    print(f"  Ingesting Raw CSV: '{raw_csv_path}'")

    raw_records = load_raw_csv(raw_csv_path)
    raw_count = len(raw_records)

    sales, invalid_count, duplicate_count = transform_and_deduplicate(raw_records)

    print(f"  Audit Summary: Raw={raw_count} | Valid={len(sales)} | Invalid={invalid_count} | Duplicates={duplicate_count}")

    print(f"  Exporting Clean Dataset to: '{processed_csv_path}'")
    export_cleaned_csv(sales, processed_csv_path)

    print(f"  Generating Analysis Report at: '{report_path}'")
    report_text = generate_sales_report(sales, raw_count, invalid_count, duplicate_count, report_path)

    print("\n[SUCCESS] Pipeline Execution Completed Successfully!")
    return sales


def display_menu():
    """Print the interactive CLI menu options."""
    print("\n==============================================")
    print("    SALES DATA PROCESSING SYSTEM — MAIN MENU  ")
    print("==============================================")
    print("1. Run Full Data Pipeline (Raw -> Clean -> Report)")
    print("2. View Dataset Audit Metrics")
    print("3. View Overall Financial Summary")
    print("4. View Highest Value Transaction")
    print("5. View Top Selling Product & Category Breakdown")
    print("6. View Generated ASCII Report")
    print("7. Exit")
    print("==============================================")


def main():
    """Main CLI execution loop."""
    base_dir = Path(__file__).resolve().parent.parent
    raw_path = base_dir / "data" / "raw" / "sales.csv"
    processed_path = base_dir / "data" / "processed" / "cleaned_sales.csv"
    report_path = base_dir / "output" / "sales_report.txt"

    # Default run pipeline on startup to ensure output files exist
    sales = run_pipeline(raw_path, processed_path, report_path)

    # Check if running interactively
    if not sys.stdin.isatty():
        print("\n[INFO] Non-interactive environment detected. Pipeline finished.")
        return

    while True:
        display_menu()
        choice = input("Select an option (1-7): ").strip()

        if choice == "1":
            sales = run_pipeline(raw_path, processed_path, report_path)
        elif choice == "2":
            raw_rec = load_raw_csv(raw_path)
            sales_clean, inv, dup = transform_and_deduplicate(raw_rec)
            print(f"\nAudit Metrics:\n  Raw Records: {len(raw_rec)}\n  Valid Clean: {len(sales_clean)}\n  Invalid: {inv}\n  Duplicates Removed: {dup}")
        elif choice == "3":
            print(f"\nFinancial Summary:\n  Total Orders: {len(sales)}\n  Total Revenue: ${total_revenue(sales):,.2f}\n  Average Order Value: ${average_order_value(sales):,.2f}")
        elif choice == "4":
            h = highest_value_order(sales)
            if h:
                print(f"\nHighest Order:\n  Order #{h.order_id} | Customer: {h.customer} | Product: {h.product} | Amount: ${h.total:,.2f}")
            else:
                print("\nNo sales available.")
        elif choice == "5":
            p, q = best_selling_product(sales)
            cats = category_revenue(sales)
            print(f"\nBest-Selling Product: {p} ({q} units sold)")
            print("\nCategory Revenue:")
            for c, r in cats.items():
                print(f"  {c:<15}: ${r:,.2f}")
        elif choice == "6":
            if report_path.exists():
                print("\n" + report_path.read_text(encoding="utf-8"))
            else:
                print("\nReport file not found. Run option 1 first.")
        elif choice == "7":
            print("\nExiting Sales Data Processing System. Goodbye!")
            break
        else:
            print("\n[ERROR] Invalid menu choice. Please select 1 to 7.")


if __name__ == "__main__":
    main()
