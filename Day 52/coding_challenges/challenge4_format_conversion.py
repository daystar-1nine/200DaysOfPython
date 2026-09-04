"""
===============================================================================
DAY 52 — CODING CHALLENGE 4: BIDIRECTIONAL CSV ↔ JSON FORMAT CONVERTER
===============================================================================
This module converts CSV data into Dataclass objects, serializes to JSON, and
converts JSON payloads back into Dataclass objects and CSV text format.
===============================================================================
"""

import csv
import io
import json
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class Item:
    """Dataclass model representing a generic item entity."""
    name: str
    marks: float


def csv_to_json(csv_text: str) -> str:
    """Convert CSV string content to JSON string via dataclass objects."""
    f = io.StringIO(csv_text.strip())
    reader = csv.DictReader(f)
    items = [Item(name=row["name"], marks=float(row["marks"])) for row in reader]
    dict_list = [asdict(item) for item in items]
    return json.dumps(dict_list, indent=4)


def json_to_csv(json_text: str) -> str:
    """Convert JSON string content to CSV text format via dataclass objects."""
    dict_list = json.loads(json_text)
    items = [Item(**d) for d in dict_list]

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=["name", "marks"])
    writer.writeheader()
    for item in items:
        writer.writerow(asdict(item))
    return out.getvalue().strip()


if __name__ == "__main__":
    sample_csv = "name,marks\nAlice,90\nBob,85"
    json_result = csv_to_json(sample_csv)
    print("CSV -> JSON Result:\n", json_result)

    csv_result = json_to_csv(json_result)
    print("JSON -> CSV Result:\n", csv_result)

    assert "Alice" in json_result and "90" in json_result
    assert "Alice,90" in csv_result or "Alice,90.0" in csv_result
    print("[OK] Challenge 4 Passed!")
