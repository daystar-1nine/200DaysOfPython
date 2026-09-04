"""
===============================================================================
DAY 52 — PRACTICE 1: NESTED UNIVERSITY JSON BUILDER & PARSER
===============================================================================
This module constructs a nested University JSON structure containing departments
and students, serializes it to disk via json.dump, reads it back via json.load,
prints names, and calculates average marks across all students.
===============================================================================
"""

import json
from pathlib import Path
from typing import Dict, Any


def create_university_data() -> Dict[str, Any]:
    """Construct nested University dictionary payload."""
    # What is used: Nested dictionary and list structure.
    # Why it is used: Demonstrates modeling complex hierarchical domain data.
    # How it works: Nests departments list and students list with subject sub-dictionaries.
    return {
        "university": {
            "name": "Global Tech University",
            "departments": ["Data Science", "Computer Science", "AI & Robotics"],
            "students": [
                {
                    "id": 1,
                    "name": "Rahul",
                    "department": "Data Science",
                    "subjects": [
                        {"name": "Python", "marks": 90},
                        {"name": "SQL", "marks": 85},
                    ],
                },
                {
                    "id": 2,
                    "name": "Aisha",
                    "department": "Computer Science",
                    "subjects": [
                        {"name": "Python", "marks": 95},
                        {"name": "Algorithms", "marks": 92},
                    ],
                },
                {
                    "id": 3,
                    "name": "Rohan",
                    "department": "AI & Robotics",
                    "subjects": [
                        {"name": "Math", "marks": 70},
                        {"name": "Python", "marks": 75},
                    ],
                },
            ],
        }
    }


def save_university_json(data: Dict[str, Any], file_path: Path) -> None:
    """Serialize dictionary data to a JSON file."""
    # What is used: pathlib.Path open context manager with json.dump(indent=4).
    # Why it is used: Writes pretty-printed JSON file to disk.
    # How it works: Opens file stream and formats JSON output.
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_and_analyze_university(file_path: Path) -> Dict[str, Any]:
    """Read university JSON file and compute student marks analytics."""
    # What is used: pathlib.Path open with json.load().
    # Why it is used: Deserializes JSON file into native Python dictionary.
    # How it works: Reads JSON file stream into data dictionary.
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    uni = data["university"]
    dept_names = uni["departments"]
    students = uni["students"]

    student_names = [s["name"] for s in students]

    # Calculate overall average marks across all student subject scores
    all_scores = [subj["marks"] for s in students for subj in s["subjects"]]
    avg_marks = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return {
        "university_name": uni["name"],
        "departments": dept_names,
        "student_names": student_names,
        "average_marks": avg_marks,
    }


if __name__ == "__main__":
    json_path = Path("Day 52/exercises/university.json")
    payload = create_university_data()
    save_university_json(payload, json_path)

    res = load_and_analyze_university(json_path)
    print("University Name:", res["university_name"])
    print("Departments    :", res["departments"])
    print("Students       :", res["student_names"])
    print(f"Average Marks  : {res['average_marks']:.2f}")

    assert res["university_name"] == "Global Tech University"
    assert len(res["departments"]) == 3
    assert len(res["student_names"]) == 3
    assert round(res["average_marks"], 2) == 84.50
    print("[OK] Practice 1 Passed!")
