"""
Day 60 - Pure Python Challenge 1: Student Statistics
Calculates total, average, highest, lowest, passing count, and above-average count without third-party libraries.
"""

# What is used: Import sys module.
# Why it is used: Configures UTF-8 console output for cross-platform stability.
# How it works: Brings sys module into execution scope.
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def calculate_student_statistics(marks: list[float]) -> dict:
    """
    Calculate summary statistics for student marks list using pure Python.

    Args:
        marks: List of numerical marks.

    Returns:
        dict: Summary statistics dictionary.
    """
    if not marks:
        return {"total": 0, "average": 0.0, "highest": 0, "lowest": 0, "passing": 0, "above_average": 0}

    # What is used: Built-in sum(), len(), max(), min().
    # Why it is used: Fast aggregation without importing NumPy or Pandas.
    # How it works: Iterates through marks list in compiled C loops.
    total = sum(marks)
    average = round(total / len(marks), 2)
    highest = max(marks)
    lowest = min(marks)

    # What is used: Generator expressions inside sum().
    # Why it is used: Counts passing students (>= 50) and students scoring above class average.
    # How it works: Evaluates boolean conditions and sums True flags.
    passing_count = sum(1 for m in marks if m >= 50)
    above_average_count = sum(1 for m in marks if m > average)

    return {
        "total": total,
        "average": average,
        "highest": highest,
        "lowest": lowest,
        "passing": passing_count,
        "above_average": above_average_count
    }


def main() -> None:
    marks = [78, 91, 65, 88, 72, 95, 54]
    stats = calculate_student_statistics(marks)

    print("==================================================")
    print("        STUDENT MARKS PURE PYTHON SUMMARY         ")
    print("==================================================")
    print(f"Marks List          : {marks}")
    print(f"Total Marks         : {stats['total']}")
    print(f"Average Mark        : {stats['average']}")
    print(f"Highest Mark        : {stats['highest']}")
    print(f"Lowest Mark         : {stats['lowest']}")
    print(f"Passing Count (>=50): {stats['passing']}")
    print(f"Above Average Count : {stats['above_average']}")


if __name__ == "__main__":
    main()
