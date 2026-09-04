"""
===============================================================================
DAY 51/52 — CODING CHALLENGE 1: JSON SERIALIZATION & DESERIALIZATION
===============================================================================
This module converts a list of student dicts to a JSON string via json.dumps()
and deserializes it back to Python using json.loads().
===============================================================================
"""

import json
from typing import List, Dict, Any


def serialize_and_deserialize(students: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Serialize list of dicts to JSON string and deserialize back to Python."""
    # What is used: json.dumps() and json.loads().
    # Why it is used: Demonstrates memory-level data serialization roundtrip.
    # How it works: Converts list of dicts to JSON string, then parses JSON string.
    json_string = json.dumps(students, indent=4)
    parsed_data = json.loads(json_string)
    return parsed_data


if __name__ == "__main__":
    sample = [
        {"name": "A", "marks": 80},
        {"name": "B", "marks": 90},
        {"name": "C", "marks": 70},
    ]
    result = serialize_and_deserialize(sample)
    print("Deserialized Result:", result)
    assert result == sample
    assert isinstance(result, list)
    assert result[0]["name"] == "A"
    print("[OK] Challenge 1 Passed!")
