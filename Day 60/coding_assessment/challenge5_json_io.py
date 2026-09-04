"""
Day 60 - Pure Python Challenge 5: JSON Serialization & Deserialization
Exports list of student dictionaries to JSON file and reads them back safely.
"""

# What is used: Import sys, json, and pathlib Path.
# Why it is used: Manages JSON file input/output and serialization.
# How it works: Serializes data using json.dump and deserializes using json.load.
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def export_and_read_students_json(students: list[dict], file_path: str | Path) -> list[dict]:
    """
    Serialize student list to JSON file on disk and deserialize it back into Python dictionaries.

    Args:
        students: List of student dictionaries.
        file_path: Target JSON file path.

    Returns:
        list[dict]: Deserialized student list read back from disk.
    """
    path = Path(file_path)

    # What is used: json.dump() with indent=2.
    # Why it is used: Serializes Python data structures into formatted UTF-8 JSON file.
    # How it works: Writes JSON text stream to file handle.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(students, f, indent=2)

    # What is used: json.load().
    # Why it is used: Deserializes JSON file contents back into Python objects.
    # How it works: Parses text stream into native Python lists and dictionaries.
    with open(path, "r", encoding="utf-8") as f:
        loaded_students = json.load(f)

    return loaded_students


def main() -> None:
    students_data = [
        {"id": 101, "name": "Aarav", "marks": 88, "course": "Computer Science"},
        {"id": 102, "name": "Diya", "marks": 94, "course": "Data Science"},
        {"id": 103, "name": "Kabir", "marks": 76, "course": "Information Tech"},
        {"id": 104, "name": "Neha", "marks": 82, "course": "Computer Science"},
        {"id": 105, "name": "Rohan", "marks": 90, "course": "Data Science"}
    ]

    target_file = Path(__file__).resolve().parent / "students.json"
    recovered = export_and_read_students_json(students_data, target_file)

    print("==================================================")
    print("           JSON FILE SERIALIZATION AUDIT          ")
    print("==================================================")
    print(f"Exported to   : {target_file}")
    print(f"Records Read  : {len(recovered)}")
    for s in recovered:
        print(f"  * ID: {s['id']} | Name: {s['name']:<8} | Marks: {s['marks']} | Course: {s['course']}")

    # Clean up created file
    if target_file.exists():
        target_file.unlink()


if __name__ == "__main__":
    main()
