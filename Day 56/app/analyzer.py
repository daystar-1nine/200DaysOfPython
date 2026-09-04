"""
Module: analyzer.py
Computes academic performance metrics, letter grades, department aggregates, and top performers.
"""

# What is used: Import pandas and numpy modules.
# Why it is used: Fundamental libraries for tabular analysis and statistics.
# How it works: Brings pandas and numpy into execution context.
import numpy as np
import pandas as pd


def analyze_student_performance(df: pd.DataFrame, pass_threshold: float = 50.0) -> tuple[pd.DataFrame, dict]:
    """
    Perform full academic performance analysis on cleaned student DataFrame.

    Args:
        df: Cleaned student DataFrame.
        pass_threshold: Minimum mark required per subject to pass (default 50.0).

    Returns:
        tuple[pd.DataFrame, dict]: Augmented DataFrame and analytical summary metrics dictionary.
    """
    # What is used: df.copy() method.
    # Why it is used: Ensures calculations do not alter original input DataFrame.
    # How it works: Duplicates memory buffer.
    analysis_df = df.copy()

    # What is used: Vectorized row sum across Math, Physics, Chemistry.
    # Why it is used: Computes total marks out of 300.
    # How it works: Adds subject Series elementwise.
    analysis_df["Total"] = analysis_df["Math"] + analysis_df["Physics"] + analysis_df["Chemistry"]

    # What is used: Vectorized division total / 3.0.
    # Why it is used: Computes overall percentage average score per student.
    # How it works: Divides Total Series by scalar 3.0.
    analysis_df["Average"] = (analysis_df["Total"] / 3.0).round(2)

    # What is used: Bitwise condition map for Pass/Fail determination.
    # Why it is used: A student passes only if scoring >= pass_threshold in ALL subjects.
    # How it works: Evaluates 3 subject conditions and maps True -> 'Pass', False -> 'Fail'.
    pass_mask = (
        (analysis_df["Math"] >= pass_threshold)
        & (analysis_df["Physics"] >= pass_threshold)
        & (analysis_df["Chemistry"] >= pass_threshold)
    )
    analysis_df["Result"] = pass_mask.map({True: "Pass", False: "Fail"})

    # What is used: pd.cut() for assigning standard letter grades based on Average score.
    # Why it is used: Categorizes numeric scores into discrete academic grade tiers.
    # How it works: Bins Average column into ranges [0-50: F, 50-70: C, 70-85: B, 85-90: A, 90-100: A+].
    bins = [0, 50, 70, 85, 90, 100]
    labels = ["F", "C", "B", "A", "A+"]
    analysis_df["Grade"] = pd.cut(analysis_df["Average"], bins=bins, labels=labels, include_lowest=True)

    # What is used: Aggregations for summary metrics.
    # Why it is used: Provides high-level insights into class performance.
    # How it works: Computes count, mean, max, and pass rates across the DataFrame.
    total_students = len(analysis_df)
    pass_count = int((analysis_df["Result"] == "Pass").sum())
    fail_count = total_students - pass_count
    pass_rate = round((pass_count / total_students * 100.0), 2) if total_students > 0 else 0.0

    overall_avg = float(analysis_df["Average"].mean()) if total_students > 0 else 0.0

    # What is used: df.loc[df['Total'].idxmax()] lookup.
    # Why it is used: Identifies the top performing overall student based on Total score.
    # How it works: idxmax() returns index label of max total, loc retrieves row Series.
    top_student = {}
    if total_students > 0:
        top_idx = analysis_df["Total"].idxmax()
        top_row = analysis_df.loc[top_idx]
        top_student = {
            "Student_ID": str(top_row["Student_ID"]),
            "Name": str(top_row["Name"]),
            "Department": str(top_row["Department"]),
            "Total": float(top_row["Total"]),
            "Average": float(top_row["Average"])
        }

    # What is used: Subject toppers via idxmax() per subject column.
    # Why it is used: Identifies highest scoring student in each subject.
    # How it works: Iterates subject names and locates row of max score.
    subject_toppers = {}
    for subj in ["Math", "Physics", "Chemistry"]:
        if total_students > 0:
            idx = analysis_df[subj].idxmax()
            row = analysis_df.loc[idx]
            subject_toppers[subj] = {
                "Name": str(row["Name"]),
                "Score": float(row[subj]),
                "Department": str(row["Department"])
            }

    # What is used: df.groupby('Department') aggregations.
    # Why it is used: Generates department-level performance metrics.
    # How it works: Groups rows by Department and computes mean scores and student counts.
    dept_summary = {}
    if total_students > 0:
        grouped = analysis_df.groupby("Department")
        for dept_name, group in grouped:
            dept_summary[str(dept_name)] = {
                "Student_Count": len(group),
                "Math_Mean": round(float(group["Math"].mean()), 2),
                "Physics_Mean": round(float(group["Physics"].mean()), 2),
                "Chemistry_Mean": round(float(group["Chemistry"].mean()), 2),
                "Overall_Avg": round(float(group["Average"].mean()), 2),
                "Pass_Rate": round(float((group["Result"] == "Pass").sum() / len(group) * 100.0), 2)
            }

    metrics = {
        "total_students": total_students,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": pass_rate,
        "overall_average": round(overall_avg, 2),
        "top_student": top_student,
        "subject_toppers": subject_toppers,
        "department_summary": dept_summary
    }

    return analysis_df, metrics
