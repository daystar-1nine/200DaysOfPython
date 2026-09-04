"""
===============================================================================
DAY 54 — MAIN APPLICATION ENTRY POINT
===============================================================================
This module executes the Student Performance Analyzer using NumPy vectorization
and provides an interactive CLI menu.
===============================================================================
"""

import sys
from pathlib import Path
from app.data import get_sample_dataset
from app.analyzer import analyze_performance
from app.report import generate_performance_report


def run_analyzer(report_path: Path):
    """Execute complete NumPy Student Performance Analyzer pipeline."""
    # What is used: Modular execution pipeline.
    # Why it is used: Ingests sample data, runs NumPy vector calculations, and writes report.
    # How it works: Calls get_sample_dataset -> analyze_performance -> generate_performance_report.
    print("\n[INFO] Running NumPy Student Performance Analyzer...")

    students, subjects, marks = get_sample_dataset()
    analysis = analyze_performance(students, subjects, marks)

    print(f"  Processed {analysis['student_count']} Students across {analysis['subject_count']} Subjects.")
    print(f"  Overall Class Average: {analysis['overall_class_average']:.2f}%")
    print(f"  Top Performer:         {analysis['best_student'][0]} ({analysis['best_student'][1]:.2f}%)")

    report_text = generate_performance_report(analysis, report_path)
    print(f"  Generated Performance Report at: '{report_path}'")

    print("\n[SUCCESS] NumPy Analyzer Execution Completed Successfully!")
    return analysis


def display_menu():
    """Print interactive CLI menu options."""
    print("\n==============================================")
    print("    NUMPY STUDENT PERFORMANCE ANALYZER MENU   ")
    print("==============================================")
    print("1. Run Full Performance Analysis & Report")
    print("2. View Student Rankings")
    print("3. View Subject Performance Averages")
    print("4. View Top & Lowest Performers")
    print("5. View Normalized Marks Matrix [0.0 - 1.0]")
    print("6. View Generated ASCII Report File")
    print("7. Exit")
    print("==============================================")


def main():
    """Main CLI execution driver."""
    base_dir = Path(__file__).resolve().parent.parent
    report_path = base_dir / "output" / "performance_report.txt"

    students, subjects, marks = get_sample_dataset()
    analysis = run_analyzer(report_path)

    if not sys.stdin.isatty():
        print("\n[INFO] Non-interactive environment detected. Analyzer finished.")
        return

    while True:
        display_menu()
        choice = input("Select an option (1-7): ").strip()

        if choice == "1":
            analysis = run_analyzer(report_path)
        elif choice == "2":
            print("\nStudent Rankings:")
            for rank, name, avg in analysis["rankings"]:
                print(f"  {rank}. {name:<12} -> {avg:.2f}%")
        elif choice == "3":
            print("\nSubject Performance Averages:")
            for subj, avg in analysis["subject_averages"].items():
                print(f"  {subj:<15} -> {avg:.2f}%")
        elif choice == "4":
            best_st, best_sc = analysis["best_student"]
            low_st, low_sc = analysis["lowest_student"]
            print(f"\nHighest Performer: {best_st} ({best_sc:.2f}%)")
            print(f"Lowest Performer:  {low_st} ({low_sc:.2f}%)")
        elif choice == "5":
            print("\nNormalized Marks Matrix [0.0 - 1.0]:")
            print(analysis["normalized_marks"].round(3))
        elif choice == "6":
            if report_path.exists():
                print("\n" + report_path.read_text(encoding="utf-8"))
            else:
                print("\nReport file not found. Run option 1 first.")
        elif choice == "7":
            print("\nExiting NumPy Student Performance Analyzer. Goodbye!")
            break
        else:
            print("\n[ERROR] Invalid menu choice. Please select 1 to 7.")


if __name__ == "__main__":
    main()
