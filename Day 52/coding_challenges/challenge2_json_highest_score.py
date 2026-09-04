"""
===============================================================================
DAY 52 — CODING CHALLENGE 2: HIGHEST SCORE FROM JSON PAYLOAD
===============================================================================
This module parses a JSON string list of student records and extracts the
student with the highest marks score.
===============================================================================
"""

import json
from typing import Tuple


def find_highest_scorer_from_json(json_payload: str) -> Tuple[str, float]:
    """Parse JSON string payload and return tuple of (highest_scorer_name, highest_marks)."""
    # What is used: json.loads() and max() with key parameter lambda.
    # Why it is used: Parses JSON payload string and extracts highest scoring student record.
    # How it works: Deserializes JSON to list of dicts and applies max(key=lambda s: s['marks']).
    data = json.loads(json_payload)
    if not data:
        raise ValueError("JSON payload is empty.")

    highest = max(data, key=lambda s: s["marks"])
    return (highest["name"], float(highest["marks"]))


if __name__ == "__main__":
    json_data = '''[
        {"name": "A", "marks": 80},
        {"name": "B", "marks": 95},
        {"name": "C", "marks": 88}
    ]'''

    name, marks = find_highest_scorer_from_json(json_data)
    print(f"Highest Scorer: {name} - {marks}")
    assert name == "B"
    assert marks == 95.0
    print("[OK] Challenge 2 Passed!")
