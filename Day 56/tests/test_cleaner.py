"""
Unit tests for data cleaner module app/cleaner.py.
"""

# What is used: Import sys module and pathlib Path class.
# Why it is used: Ensures app package imports resolve cleanly during pytest execution.
# How it works: Appends Day 56 parent path to sys.path.
import sys
from pathlib import Path

DAY56_DIR = Path(__file__).resolve().parent.parent
if str(DAY56_DIR) not in sys.path:
    sys.path.insert(0, str(DAY56_DIR))

# What is used: Import pandas and cleaner module functions.
# Why it is used: Tests data cleaning pipeline, duplicate removal, and NaN imputation.
# How it works: Executes clean_student_data on test data.
import pandas as pd
from app.cleaner import clean_student_data


def test_clean_duplicates_removed(sample_raw_df):
    """
    Test duplicate Student_ID records are removed.
    """
    # What is used: clean_student_data call on sample DataFrame with duplicates.
    # Why it is used: Verifies duplicate removal logic.
    # How it works: Checks that duplicates_removed counter is 1 and resulting row count is 4.
    cleaned_df, stats = clean_student_data(sample_raw_df)
    assert stats["duplicates_removed"] == 1
    assert len(cleaned_df) == 4
    assert cleaned_df["Student_ID"].tolist() == ["S101", "S102", "S103", "S104"]


def test_clean_null_imputation(sample_raw_df):
    """
    Test missing numerical scores are imputed using subject median.
    """
    # What is used: clean_student_data execution.
    # Why it is used: Ensures zero NaNs remain after cleaning pipeline runs.
    # How it works: Asserts total null count in subject columns is 0.
    cleaned_df, stats = clean_student_data(sample_raw_df)
    assert stats["nulls_filled"] > 0
    assert cleaned_df[["Math", "Physics", "Chemistry"]].isnull().sum().sum() == 0


def test_clean_string_stripping(sample_raw_df):
    """
    Test leading/trailing whitespace is stripped from text columns.
    """
    # What is used: clean_student_data execution on whitespace-padded strings.
    # Why it is used: Asserts string fields are cleanly stripped.
    # How it works: Compares Name and Department values against stripped targets.
    cleaned_df, _ = clean_student_data(sample_raw_df)
    assert cleaned_df.loc[cleaned_df["Student_ID"] == "S101", "Name"].values[0] == "Aarav Sharma"
    assert cleaned_df.loc[cleaned_df["Student_ID"] == "S102", "Name"].values[0] == "Ananya Patel"


def test_clean_invalid_scores_dropped():
    """
    Test out-of-bounds scores (< 0 or > 100) are dropped.
    """
    # What is used: DataFrame containing negative mark and mark > 100.
    # Why it is used: Verifies boundary filtering.
    # How it works: Confirms invalid score rows are dropped from cleaned DataFrame.
    invalid_data = pd.DataFrame({
        "Student_ID": ["S1", "S2", "S3"],
        "Name": ["A", "B", "C"],
        "Department": ["CSE", "DS", "ECE"],
        "Math": [85.0, -10.0, 90.0],
        "Physics": [90.0, 80.0, 150.0],
        "Chemistry": [92.0, 85.0, 88.0]
    })
    cleaned_df, stats = clean_student_data(invalid_data)
    assert stats["invalid_scores_dropped"] == 2
    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["Student_ID"] == "S1"
