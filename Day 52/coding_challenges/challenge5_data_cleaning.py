"""
===============================================================================
DAY 52 — CODING CHALLENGE 5: DATA CLEANING & TYPE CONVERSION PIPELINE
===============================================================================
This module cleans raw messy dictionary records (whitespace removal, string-to-int
conversion, string-to-float conversion) into structured typed records.
===============================================================================
"""

from typing import List, Dict, Any


def clean_student_data(raw_records: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Clean and transform raw messy string records into typed dictionaries."""
    # What is used: String strip() and explicit type casting.
    # Why it is used: Sanitizes leading/trailing whitespace and converts age to int, marks to float.
    # How it works: Iterates raw records, strips string keys, casts numeric values.
    cleaned: List[Dict[str, Any]] = []
    for record in raw_records:
        clean_name = record["name"].strip()
        clean_age = int(record["age"].strip())
        clean_marks = float(record["marks"].strip())
        cleaned.append({
            "name": clean_name,
            "age": clean_age,
            "marks": clean_marks,
        })
    return cleaned


if __name__ == "__main__":
    messy_input = [
        {"name": " Rahul ", "age": "21", "marks": "85"},
        {"name": "Aisha", "age": "20", "marks": "92"},
    ]

    cleaned_output = clean_student_data(messy_input)
    print("Cleaned Data Output:", cleaned_output)
    expected = [
        {"name": "Rahul", "age": 21, "marks": 85.0},
        {"name": "Aisha", "age": 20, "marks": 92.0},
    ]
    assert cleaned_output == expected
    assert isinstance(cleaned_output[0]["age"], int)
    assert isinstance(cleaned_output[0]["marks"], float)
    print("[OK] Challenge 5 Passed!")
