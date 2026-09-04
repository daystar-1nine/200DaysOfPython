"""
Unit tests for analytical engine module app/analyzer.py.
"""

# What is used: Import sys module and pathlib Path class.
# Why it is used: Ensures app package imports resolve cleanly during pytest execution.
# How it works: Appends Day 56 parent path to sys.path.
import sys
from pathlib import Path

DAY56_DIR = Path(__file__).resolve().parent.parent
if str(DAY56_DIR) not in sys.path:
    sys.path.insert(0, str(DAY56_DIR))

# What is used: Import pandas and analyzer functions.
# Why it is used: Asserts accuracy of grade binning, pass/fail logic, top performer, and department stats.
# How it works: Executes analyze_student_performance on test dataset.
import pandas as pd
from app.analyzer import analyze_student_performance


def test_analyze_grade_assignment():
    """
    Test grade assignment and average calculation.
    """
    # What is used: Prepared clean test DataFrame.
    # Why it is used: Validates expected Grade and Average outputs.
    # How it works: Compares calculated Grade column values against expected letter grades.
    test_df = pd.DataFrame({
        "Student_ID": ["S101", "S102", "S103"],
        "Name": ["Aarav", "Ananya", "Kabir"],
        "Department": ["CSE", "DS", "ECE"],
        "Math": [95.0, 80.0, 40.0],
        "Physics": [90.0, 75.0, 45.0],
        "Chemistry": [94.0, 78.0, 42.0]
    })

    analysis_df, metrics = analyze_student_performance(test_df)

    assert "Total" in analysis_df.columns
    assert "Average" in analysis_df.columns
    assert "Grade" in analysis_df.columns
    assert "Result" in analysis_df.columns

    assert analysis_df.loc[0, "Grade"] == "A+"
    assert analysis_df.loc[1, "Grade"] == "B"
    assert analysis_df.loc[2, "Grade"] == "F"


def test_analyze_pass_fail_metrics():
    """
    Test pass count, fail count, and pass percentage calculations.
    """
    # What is used: Test DataFrame with passing and failing students.
    # Why it is used: Verifies overall class summary metrics dictionary.
    # How it works: Asserts total_students, pass_count, fail_count, and pass_rate values.
    test_df = pd.DataFrame({
        "Student_ID": ["S1", "S2", "S3", "S4"],
        "Name": ["A", "B", "C", "D"],
        "Department": ["CSE", "CSE", "DS", "DS"],
        "Math": [80.0, 40.0, 70.0, 90.0],
        "Physics": [85.0, 60.0, 75.0, 95.0],
        "Chemistry": [90.0, 70.0, 80.0, 92.0]
    })

    _, metrics = analyze_student_performance(test_df, pass_threshold=50.0)

    assert metrics["total_students"] == 4
    assert metrics["pass_count"] == 3
    assert metrics["fail_count"] == 1
    assert metrics["pass_rate"] == 75.0


def test_analyze_top_student_and_subject_toppers():
    """
    Test top overall performer and subject toppers identification.
    """
    # What is used: Test DataFrame with known top performers.
    # Why it is used: Verifies idxmax lookup accuracy for overall and subject toppers.
    # How it works: Asserts top student name and subject toppers scores.
    test_df = pd.DataFrame({
        "Student_ID": ["S1", "S2", "S3"],
        "Name": ["Aarav", "Ananya", "Rohan"],
        "Department": ["CSE", "DS", "ECE"],
        "Math": [98.0, 85.0, 70.0],
        "Physics": [80.0, 99.0, 60.0],
        "Chemistry": [85.0, 88.0, 96.0]
    })

    _, metrics = analyze_student_performance(test_df)

    assert metrics["top_student"]["Name"] == "Ananya"  # Total 272 vs Aarav 263
    assert metrics["subject_toppers"]["Math"]["Name"] == "Aarav"
    assert metrics["subject_toppers"]["Physics"]["Name"] == "Ananya"
    assert metrics["subject_toppers"]["Chemistry"]["Name"] == "Rohan"


def test_analyze_department_breakdown():
    """
    Test department summary statistics calculation.
    """
    # What is used: Multi-department test DataFrame.
    # Why it is used: Verifies group aggregation metrics per department.
    # How it works: Asserts student count and averages per department key.
    test_df = pd.DataFrame({
        "Student_ID": ["S1", "S2", "S3"],
        "Name": ["A", "B", "C"],
        "Department": ["CSE", "CSE", "DS"],
        "Math": [80.0, 90.0, 70.0],
        "Physics": [80.0, 90.0, 70.0],
        "Chemistry": [80.0, 90.0, 70.0]
    })

    _, metrics = analyze_student_performance(test_df)
    dept_summary = metrics["department_summary"]

    assert "CSE" in dept_summary
    assert "DS" in dept_summary
    assert dept_summary["CSE"]["Student_Count"] == 2
    assert dept_summary["CSE"]["Overall_Avg"] == 85.0
    assert dept_summary["DS"]["Student_Count"] == 1
    assert dept_summary["DS"]["Overall_Avg"] == 70.0
