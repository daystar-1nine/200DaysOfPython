# ==============================================================================
# Program    : Main Data Transformation Pipeline (main_pipeline.py)
# Objective  : Execute functional pipeline (Raw -> Validate -> Filter -> Transform -> Sort -> Aggregate).
# Concept    : Functional Composition Pipeline
# Why Used   : Connects functional stages into an end-to-end data processing workflow.
# ==============================================================================

import os
import sys

pkg_root = os.path.abspath(os.path.dirname(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from pipeline.validator import validate_transactions
from pipeline.filters import filter_high_value
from pipeline.transformer import transform_records
from pipeline.sorter import sort_by_key
from pipeline.aggregator import generate_category_report

def run_pipeline(raw_data: list[dict], min_amount: float = 500.0) -> dict:
    """Executes pure functional data pipeline."""
    # 1. Validate
    valid = validate_transactions(raw_data)
    # 2. Filter
    filtered = filter_high_value(valid, min_amount=min_amount)
    # 3. Transform
    transformed = transform_records(filtered)
    # 4. Sort
    sorted_recs = sort_by_key(transformed, key="amount", reverse=True)
    # 5. Aggregate & Report
    report = generate_category_report(sorted_recs)
    report["records"] = sorted_recs
    return report

def main():
    raw_sample = [
        {"id": 1, "amount": 1200.0, "category": " travel "},
        {"id": 2, "amount": 250.0, "category": "Food"},
        {"id": 3, "amount": -50.0, "category": "Invalid"},  # Should be filtered out by validation
        {"id": 4, "amount": 850.0, "category": "Shopping"},
        {"id": 5, "amount": 1500.0, "category": "TRAVEL"}
    ]

    print("==================================================")
    print("     DAY 38 - FUNCTIONAL DATA PIPELINE DEMO       ")
    print("==================================================\n")

    res = run_pipeline(raw_sample, min_amount=500.0)

    print("--- 1. Filtered & Transformed Records (Sorted Descending) ---")
    for r in res["records"]:
        print(f"ID #{r['id']}: {r['category']:<12} -> {r['formatted_amount']}")

    print("\n--- 2. Aggregated Category Breakdown ---")
    for cat, amt in res["category_breakdown"].items():
        print(f"  {cat:<12}: Rs.{amt:,.2f}")

    print(f"\nGrand Total           : Rs.{res['grand_total']:,.2f}")
    print(f"Has Large Tx (>=1000) : {res['has_large_transaction']}")
    print(f"All Positive Amounts  : {res['all_positive']}")

if __name__ == "__main__":
    main()
