"""
Pytest configuration file and reusable test fixtures for Day 56.
"""

# What is used: Import pytest and pandas modules.
# Why it is used: Provides fixture declaration decorators and sample DataFrame construction.
# How it works: Registers fixtures in pytest context.
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """
    Fixture providing a raw sample DataFrame with missing values and duplicates.
    """
    # What is used: Dictionary containing mock uncleaned student dataset.
    # Why it is used: Simulates raw input dataset for testing cleaning and loader modules.
    # How it works: Constructs DataFrame with intentional missing values and repeated ID.
    data = {
        "Student_ID": ["S101", "S102", "S103", "S101", "S104"],
        "Name": ["Aarav Sharma ", " Ananya Patel", "Rohan Verma", "Aarav Sharma ", "Priya Nair"],
        "Department": ["CSE", "DS", "ECE", "CSE", "CSE"],
        "Math": [85.0, 92.0, None, 85.0, 45.0],
        "Physics": [90.0, None, 82.0, 90.0, 52.0],
        "Chemistry": [92.0, 95.0, 80.0, 92.0, 48.0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(tmp_path, sample_raw_df) -> str:
    """
    Fixture providing a temporary CSV file path populated with raw student data.
    """
    # What is used: tmp_path fixture provided by pytest.
    # Why it is used: Creates isolated temporary file for filesystem I/O testing.
    # How it works: Writes DataFrame to temp CSV path and returns string path.
    csv_file = tmp_path / "test_students.csv"
    sample_raw_df.to_csv(csv_file, index=False)
    return str(csv_file)
