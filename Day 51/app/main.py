"""
===============================================================================
DAY 51 — STUDENT DATA PROCESSOR CLI (MAIN ENTRY POINT)
===============================================================================
This module serves as the CLI interface entry point, handling menu selection,
CSV loading via file_handler, analysis execution, and report generation.
===============================================================================
"""

import sys
from pathlib import Path
from app.file_handler import read_students_csv
from app.services import (
    calculate_average,
    get_highest_scorer,
    get_lowest_scorer,
    count_students,
    filter_students_by_marks,
)
from app.reports import generate_student_report
from app.utils import format_student_table


def run_cli(csv_path: Path, output_path: Path) -> None:
    """Run interactive Student Data Processor CLI application loop."""
    print("========================================")
    print("   STUDENT DATA PROCESSOR CLI (DAY 51)  ")
    print("========================================")

    try:
        students = read_students_csv(csv_path)
        print(f"Loaded {len(students)} student records from '{csv_path}'.\n")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    while True:
        print("\n--- MENU OPTIONS ---")
        print("1. Show all students")
        print("2. Calculate average marks")
        print("3. Find highest scorer")
        print("4. Find lowest scorer")
        print("5. Count students")
        print("6. Filter students by marks")
        print("7. Generate report file")
        print("8. Exit")

        try:
            choice = input("Enter choice (1-8): ").strip()
        except EOFError:
            print("\nExiting CLI.")
            break

        if choice == "1":
            print("\n" + format_student_table(students))
        elif choice == "2":
            avg = calculate_average(students)
            print(f"\nAverage Marks: {avg:.2f}")
        elif choice == "3":
            highest = get_highest_scorer(students)
            print(f"\nHighest Scorer: {highest.name} - {highest.marks:.2f} ({highest.performance_level.value})")
        elif choice == "4":
            lowest = get_lowest_scorer(students)
            print(f"\nLowest Scorer: {lowest.name} - {lowest.marks:.2f} ({lowest.performance_level.value})")
        elif choice == "5":
            print(f"\nTotal Students: {count_students(students)}")
        elif choice == "6":
            try:
                min_m = float(input("Enter minimum marks threshold: ").strip())
                filtered = filter_students_by_marks(students, min_m)
                print(f"\nFound {len(filtered)} student(s) with marks >= {min_m}:")
                print(format_student_table(filtered))
            except ValueError as e:
                print(f"Invalid input threshold: {e}")
        elif choice == "7":
            report_text = generate_student_report(students, output_path)
            print(f"\nReport generated and saved to '{output_path}'.\n")
            print(report_text)
        elif choice == "8":
            print("Thank you for using Student Data Processor CLI. Exiting!")
            break
        else:
            print("Invalid selection. Please choose an option between 1 and 8.")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    raw_csv = base_dir / "data" / "raw" / "students.csv"
    out_file = base_dir / "output" / "report.txt"
    run_cli(raw_csv, out_file)
