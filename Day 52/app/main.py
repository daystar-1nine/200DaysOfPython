"""
===============================================================================
DAY 52 — STUDENT DATA MANAGEMENT SYSTEM V2 (MAIN CLI ENTRY POINT)
===============================================================================
This module provides the interactive CLI application loop for Student Data
Management System V2, supporting JSON/CSV import/export, student CRUD, statistics,
and report generation.
===============================================================================
"""

from pathlib import Path
from typing import List
from app.models import Student
from app.json_handler import save_students, load_students
from app.csv_handler import save_students_csv, load_students_csv
from app.services import (
    generate_next_id,
    add_student,
    update_student,
    delete_student,
    search_students,
    calculate_average,
    get_highest_scorer,
    get_lowest_scorer,
)
from app.reports import generate_student_report
from app.utils import format_student_table


def run_cli(json_path: Path, csv_path: Path, output_path: Path) -> None:
    """Run interactive Student Data Management System V2 CLI loop."""
    print("==================================================")
    print("   STUDENT DATA MANAGEMENT SYSTEM V2 (DAY 52)     ")
    print("==================================================")

    students: List[Student] = []

    # Attempt loading initial data from JSON, fallback to CSV
    if json_path.exists():
        try:
            students = load_students(json_path)
            print(f"Loaded {len(students)} student record(s) from JSON ('{json_path}').")
        except Exception as e:
            print(f"Notice: Failed loading JSON ('{json_path}'): {e}")
    elif csv_path.exists():
        try:
            students = load_students_csv(csv_path)
            print(f"Loaded {len(students)} student record(s) from CSV ('{csv_path}').")
        except Exception as e:
            print(f"Notice: Failed loading CSV ('{csv_path}'): {e}")

    while True:
        print("\n--- MENU OPTIONS ---")
        print("1.  Add Student")
        print("2.  View Students")
        print("3.  Search Student (ID, Name, or Course)")
        print("4.  Update Student")
        print("5.  Delete Student")
        print("6.  Calculate Average Marks")
        print("7.  Find Highest Scorer")
        print("8.  Find Lowest Scorer")
        print("9.  Export to JSON")
        print("10. Export to CSV")
        print("11. Import from JSON")
        print("12. Import from CSV")
        print("13. Generate Report File")
        print("14. Exit")

        try:
            choice = input("Enter choice (1-14): ").strip()
        except EOFError:
            print("\nExiting application.")
            break

        if choice == "1":
            try:
                name = input("Enter student name: ").strip()
                age = int(input("Enter student age: ").strip())
                course = input("Enter course name: ").strip()
                marks = float(input("Enter student marks (0-100): ").strip())
                new_id = generate_next_id(students)
                new_s = Student(id=new_id, name=name, age=age, course=course, marks=marks)
                add_student(students, new_s)
                print(f"\nStudent '{name}' added successfully with assigned ID {new_id}.")
            except ValueError as e:
                print(f"\nFailed to add student: {e}")

        elif choice == "2":
            print("\n" + format_student_table(students))

        elif choice == "3":
            q = input("Enter search query (ID, Name, or Course): ").strip()
            results = search_students(students, q)
            print(f"\nFound {len(results)} matching record(s):")
            print(format_student_table(results))

        elif choice == "4":
            try:
                s_id = int(input("Enter student ID to update: ").strip())
                print("Leave blank to keep existing value.")
                new_name = input("New name: ").strip() or None
                age_str = input("New age: ").strip()
                new_age = int(age_str) if age_str else None
                new_course = input("New course: ").strip() or None
                marks_str = input("New marks: ").strip()
                new_marks = float(marks_str) if marks_str else None

                updated = update_student(
                    students,
                    student_id=s_id,
                    name=new_name,
                    age=new_age,
                    course=new_course,
                    marks=new_marks,
                )
                print(f"\nStudent ID {s_id} updated successfully: {updated.name}")
            except (ValueError, KeyError) as e:
                print(f"\nFailed to update student: {e}")

        elif choice == "5":
            try:
                s_id = int(input("Enter student ID to delete: ").strip())
                deleted = delete_student(students, s_id)
                print(f"\nStudent ID {s_id} ('{deleted.name}') deleted successfully.")
            except (ValueError, KeyError) as e:
                print(f"\nFailed to delete student: {e}")

        elif choice == "6":
            avg = calculate_average(students)
            print(f"\nAverage Marks: {avg:.2f}")

        elif choice == "7":
            try:
                highest = get_highest_scorer(students)
                print(f"\nHighest Scorer: {highest.name} - {highest.marks:.2f} ({highest.course})")
            except ValueError as e:
                print(f"\nError: {e}")

        elif choice == "8":
            try:
                lowest = get_lowest_scorer(students)
                print(f"\nLowest Scorer: {lowest.name} - {lowest.marks:.2f} ({lowest.course})")
            except ValueError as e:
                print(f"\nError: {e}")

        elif choice == "9":
            try:
                save_students(students, json_path)
                print(f"\nSuccessfully exported {len(students)} record(s) to JSON: '{json_path}'.")
            except Exception as e:
                print(f"\nExport to JSON failed: {e}")

        elif choice == "10":
            try:
                save_students_csv(students, csv_path)
                print(f"\nSuccessfully exported {len(students)} record(s) to CSV: '{csv_path}'.")
            except Exception as e:
                print(f"\nExport to CSV failed: {e}")

        elif choice == "11":
            try:
                students = load_students(json_path)
                print(f"\nSuccessfully imported {len(students)} record(s) from JSON: '{json_path}'.")
            except Exception as e:
                print(f"\nImport from JSON failed: {e}")

        elif choice == "12":
            try:
                students = load_students_csv(csv_path)
                print(f"\nSuccessfully imported {len(students)} record(s) from CSV: '{csv_path}'.")
            except Exception as e:
                print(f"\nImport from CSV failed: {e}")

        elif choice == "13":
            try:
                report_text = generate_student_report(students, output_path)
                print(f"\nReport generated and saved to '{output_path}':\n")
                print(report_text)
            except Exception as e:
                print(f"\nReport generation failed: {e}")

        elif choice == "14":
            # Auto-save changes to JSON before exiting
            try:
                save_students(students, json_path)
                print("Data saved to JSON.")
            except Exception:
                pass
            print("Exiting Student Data Management System V2. Goodbye!")
            break
        else:
            print("Invalid selection. Please choose an option between 1 and 14.")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    j_path = base_dir / "data" / "students.json"
    c_path = base_dir / "data" / "students.csv"
    o_path = base_dir / "output" / "student_report.txt"
    run_cli(j_path, c_path, o_path)
