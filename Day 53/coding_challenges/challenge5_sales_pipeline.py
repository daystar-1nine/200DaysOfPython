"""
===============================================================================
DAY 53 — CODING CHALLENGE 5: COMPLETE SALES PIPELINE
===============================================================================
Topic: End-to-End Modular Data Processing Pipeline Function
Goal: Load raw CSV -> Clean -> Validate -> Deduplicate -> Transform -> Analyze
===============================================================================
"""

import csv
from pathlib import Path
from typing import Dict, Any, List


def process_sales(csv_path: Path) -> Dict[str, Any]:
    """Execute complete modular data processing pipeline on a raw sales CSV file."""
    # What is used: Modular functional pipeline steps (Load, Clean, Validate, Deduplicate, Analyze).
    # Why it is used: Demonstrates separation of concerns in pure Python data processing.
    # How it works: Reads CSV file, normalizes fields, filters invalid entries, deduplicates by ID,
    #               computes totals, and returns analytical metrics dict.
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV file not found at: {csv_path}")

    # Stage 1: Load Raw CSV Records
    raw_records: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_records = list(reader)

    # Stage 2 & 3: Clean & Validate Records
    cleaned_valid: List[Dict[str, Any]] = []
    invalid_count = 0

    for row in raw_records:
        try:
            order_id = int(row["order_id"].strip())
            customer = row["customer"].strip().title()
            product = row["product"].strip().title()
            category = row["category"].strip().title()
            price = float(row["price"].strip())
            quantity = int(row["quantity"].strip())
            date_str = row["date"].strip()

            if order_id <= 0 or price < 0 or quantity <= 0 or not customer or not product:
                invalid_count += 1
                continue

            cleaned_valid.append({
                "order_id": order_id,
                "customer": customer,
                "product": product,
                "category": category,
                "price": price,
                "quantity": quantity,
                "date": date_str,
                "total": price * quantity
            })
        except (ValueError, KeyError):
            invalid_count += 1

    # Stage 4: Deduplicate Records by order_id
    seen_ids = set()
    deduplicated: List[Dict[str, Any]] = []
    duplicate_count = 0

    for rec in cleaned_valid:
        if rec["order_id"] in seen_ids:
            duplicate_count += 1
        else:
            seen_ids.add(rec["order_id"])
            deduplicated.append(rec)

    # Stage 5: Analytical Summary Computations
    total_orders = len(deduplicated)
    total_revenue = sum(r["total"] for r in deduplicated)
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0.0

    return {
        "raw_count": len(raw_records),
        "valid_count": len(cleaned_valid),
        "invalid_count": invalid_count,
        "duplicate_count": duplicate_count,
        "processed_count": total_orders,
        "total_revenue": total_revenue,
        "average_order_value": avg_order_value,
        "records": deduplicated
    }


if __name__ == "__main__":
    import tempfile

    # Create temporary raw CSV file for testing
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", encoding="utf-8") as f:
        f.write("order_id,customer,product,category,price,quantity,date\n")
        f.write("1001, rahul ,laptop,electronics,55000,1,2026-09-01\n")
        f.write("1002,aisha,mouse,electronics,1200,2,2026-09-01\n")
        f.write("1002,aisha,mouse,electronics,1200,2,2026-09-01\n")  # duplicate
        f.write("1003,rohan,chair,furniture,-500,1,2026-09-02\n")     # invalid price
        temp_path = Path(f.name)

    try:
        report = process_sales(temp_path)
        print("Pipeline Execution Summary:")
        print("  Raw Records:      ", report["raw_count"])
        print("  Processed Records:", report["processed_count"])
        print("  Invalid Records:  ", report["invalid_count"])
        print("  Duplicates:       ", report["duplicate_count"])
        print(f"  Total Revenue:     ${report['total_revenue']:,.2f}")

        assert report["raw_count"] == 4, "Raw count failed"
        assert report["processed_count"] == 2, "Processed count failed"
        assert report["invalid_count"] == 1, "Invalid count failed"
        assert report["duplicate_count"] == 1, "Duplicate count failed"
        assert report["total_revenue"] == 57400.0, "Revenue calculation failed"
        print("[OK] Challenge 5 Passed Successfully!")
    finally:
        if temp_path.exists():
            temp_path.unlink()
