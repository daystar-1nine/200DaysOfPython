"""
===============================================================================
DAY 52 — CODING CHALLENGE 3: CSV READING & AVERAGE SCORE CALCULATION
===============================================================================
This module parses a CSV string with header 'name,marks' using csv.DictReader,
converts string marks to floats, and calculates the mean average score.
===============================================================================
"""

import csv
import io


def calculate_csv_average_marks(csv_content: str) -> float:
    """Parse CSV text content and calculate average student marks."""
    # What is used: io.StringIO and csv.DictReader.
    # Why it is used: Parses string as file stream converting cell strings to numeric floats.
    # How it works: Iterates DictReader rows, extracts float(row['marks']), and averages sum.
    f = io.StringIO(csv_content.strip())
    reader = csv.DictReader(f)

    scores = [float(row["marks"]) for row in reader]
    if not scores:
        return 0.0

    return sum(scores) / len(scores)


if __name__ == "__main__":
    raw_csv = """name,marks
A,80
B,95
C,88"""

    avg = calculate_csv_average_marks(raw_csv)
    print(f"Calculated CSV Average: {avg:.2f}")
    assert round(avg, 2) == 87.67
    print("[OK] Challenge 3 Passed!")
