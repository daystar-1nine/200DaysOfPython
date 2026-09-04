"""
===============================================================================
DAY 55 — MAIN APPLICATION ENTRY POINT
===============================================================================
This module runs the Advanced Student Analytics Engine V2 using NumPy
and provides an interactive CLI menu.
===============================================================================
"""

import sys
from pathlib import Path
from app.generator import generate_student_dataset
from app.analyzer import analyze_student_performance_v2
from app.report import generate_analytics_report


def run_analytics_engine(report_path: Path):
    """Execute complete Student Analytics Engine V2 pipeline."""
    # What is used: Modular execution pipeline.
    # Why it is used: Generates reproducible synthetic data with NaNs, runs NumPy calculations, and saves report.
    # How it works: Calls generate_student_dataset -> analyze_student_performance_v2 -> generate_analytics_report.
    print("\n[INFO] Starting Advanced Student Analytics Engine V2...")

    students, subjects, marks = generate_student_dataset(num_students=100, seed=42, insert_nans=True)
    analysis = analyze_student_performance_v2(students, subjects, marks)

    print(f"  Processed {analysis['student_count']} Students across {analysis['subject_count']} Subjects.")
    print(f"  Overall Class Average: {analysis['overall_class_average']:.2f}%")
    print(f"  Pass Percentage:       {analysis['pass_fail']['pass_percentage']:.1f}%")
    print(f"  Top Student:           {analysis['best_student'][0]} ({analysis['best_student'][1]:.2f}%)")

    report_text = generate_analytics_report(analysis, report_path)
    print(f"  Generated Analytics Report at: '{report_path}'")

    print("\n[SUCCESS] Analytics Engine Execution Completed Successfully!")
    return analysis, students, subjects, marks


def display_menu():
    """Print interactive CLI menu options."""
    print("\n==============================================")
    print("    STUDENT ANALYTICS ENGINE V2 — MENU       ")
    print("==============================================")
    print("1. Run Full Analytics Pipeline & Generate Report")
    print("2. View Class Overview & Pass/Fail Metrics")
    print("3. View Top 10 Student Rankings")
    print("4. View Subject Performance Breakdown")
    print("5. View Letter Grade Distribution")
    print("6. View Generated ASCII Report File")
    print("7. Exit")
    print("==============================================")


def main():
    """Main CLI driver."""
    base_dir = Path(__file__).resolve().parent.parent
    report_path = base_dir / "output" / "student_analytics_report.txt"

    analysis, students, subjects, marks = run_analytics_engine(report_path)

    if not sys.stdin.isatty():
        print("\n[INFO] Non-interactive environment detected. Engine finished.")
        return

    while True:
        display_menu()
        choice = input("Select an option (1-7): ").strip()

        if choice == "1":
            analysis, students, subjects, marks = run_analytics_engine(report_path)
        elif choice == "2":
            pf = analysis["pass_fail"]
            print(f"\nClass Overview:\n  Class Average:   {analysis['overall_class_average']:.2f}%\n  Passed Students: {pf['pass_count']} ({pf['pass_percentage']:.1f}%)\n  Failed Students: {pf['fail_count']} ({pf['fail_percentage']:.1f}%)")
        elif choice == "3":
            print("\nTop 10 Students:")
            for rank, name, avg in analysis["top_10_students"]:
                print(f"  {rank:2d}. {name:<12} -> {avg:.2f}%")
        elif choice == "4":
            print("\nSubject Performance Breakdown:")
            for subj, avg in analysis["subject_averages"].items():
                print(f"  {subj:<15} -> {avg:.2f}%")
        elif choice == "5":
            print("\nGrade Distribution:")
            for grade, count in analysis["grade_distribution"].items():
                print(f"  Grade {grade:<3} -> {count} students")
        elif choice == "6":
            if report_path.exists():
                print("\n" + report_path.read_text(encoding="utf-8"))
            else:
                print("\nReport file not found. Run option 1 first.")
        elif choice == "7":
            print("\nExiting Student Analytics Engine V2. Goodbye!")
            break
        else:
            print("\n[ERROR] Invalid menu choice. Please select 1 to 7.")


if __name__ == "__main__":
    main()
