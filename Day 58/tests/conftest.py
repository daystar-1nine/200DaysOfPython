"""
Pytest configuration file and reusable test fixtures for Day 58.
"""

# What is used: Import pytest and pandas modules.
# Why it is used: Declares pytest fixture decorators and sample DataFrame construction.
# How it works: Registers fixtures in test environment.
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_messy_df() -> pd.DataFrame:
    """
    Fixture providing a raw messy customer DataFrame with missing values, duplicate Customer_IDs, extra spaces, and invalid numbers.
    """
    # What is used: Dictionary containing mock uncleaned customer data.
    # Why it is used: Simulates raw input dataset for loader, cleaner, validator, and analyzer tests.
    # How it works: Holds messy strings, invalid age (-5), currency symbol (₹60,000), duplicate ID C101.
    data = {
        "Customer_ID": ["C101", "C102", "C101", "C103", "C104"],
        "Name": [" Rahul Sharma ", "Priya Patel", " Rahul Sharma ", "Aman Verma", "Sneha Kulkarni"],
        "Age": ["28", "25", "28", "-5", "30"],
        "Gender": ["M", "female", "M", "MALE", "F"],
        "Email": ["rahul@example.com", "priya@domain.com", "rahul@example.com", None, "sneha@domain.com"],
        "Phone": ["98765-43210", "9876543210", "98765-43210", "+91 9876543211", "9876543212"],
        "City": [" mumbai ", "PUNE", " mumbai ", "DELHI", "Mumbai"],
        "Salary": ["₹60,000", "₹55,000", "₹60,000", "unknown", "₹75,000"],
        "Join_Date": ["01-01-2026", "2026/01/05", "01-01-2026", "March 10, 2026", "invalid_date"],
        "Department": [" Engineering ", "Data Science", " Engineering ", "Hardware", "Marketing"]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(tmp_path, sample_raw_messy_df) -> str:
    """
    Fixture creating a temporary CSV file populated with raw messy customer data.
    """
    # What is used: tmp_path fixture provided by pytest.
    # Why it is used: Isolates file reading tests within temporary filesystem directory.
    # How it works: Saves sample DataFrame to CSV and returns file path string.
    file_p = tmp_path / "test_messy_customers.csv"
    sample_raw_messy_df.to_csv(file_p, index=False)
    return str(file_p)


@pytest.fixture
def sample_clean_customer_df() -> pd.DataFrame:
    """
    Fixture providing a clean processed customer DataFrame.
    """
    # What is used: Dictionary containing valid, standardized customer data.
    # Why it is used: Input data for validator and analyzer unit tests.
    # How it works: Provides clean customer records with float Age/Salary and datetime Join_Date.
    data = {
        "customer_id": ["C101", "C102", "C103", "C104"],
        "name": ["Rahul Sharma", "Priya Patel", "Aman Verma", "Sneha Kulkarni"],
        "age": [28.0, 25.0, 28.0, 30.0],
        "gender": ["Male", "Female", "Male", "Female"],
        "email": ["rahul@example.com", "priya@domain.com", "unknown@example.com", "sneha@domain.com"],
        "phone": ["9876543210", "9876543210", "9876543211", "9876543212"],
        "city": ["Mumbai", "Pune", "Delhi", "Mumbai"],
        "salary": [60000.0, 55000.0, 60000.0, 75000.0],
        "join_date": pd.to_datetime(["2026-01-01", "2026-01-05", "2026-03-10", "2026-01-12"]),
        "department": ["Engineering", "Data Science", "Hardware", "Marketing"],
        "join_year": [2026, 2026, 2026, 2026],
        "join_month": [1, 1, 3, 1],
        "join_month_name": ["January", "January", "March", "January"]
    }
    return pd.DataFrame(data)
