"""
Day 58 - Practical Task 6: Domain Range Validation & Outlier Filtering
Demonstrates validating student records to detect invalid Age (< 0, > 100) and Marks (< 0, > 100).
"""

# What is used: Import pandas library.
# Why it is used: Core package for boolean range filtering and validation.
# How it works: Brings pandas namespace into execution context.
import pandas as pd


def validate_student_records(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate student records against business rules: Age (0-100) and Marks (0-100).

    Args:
        df: Input student DataFrame.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Filtered valid DataFrame and invalid records DataFrame.
    """
    # What is used: Boolean range indexing df["Age"].between(0, 100) & df["Marks"].between(0, 100).
    # Why it is used: Filters rows residing within legitimate academic boundaries.
    # How it works: Evaluates two range conditions elementwise.
    valid_mask = df["Age"].between(0, 100) & df["Marks"].between(0, 100)

    # What is used: Bracket indexing df[valid_mask] and df[~valid_mask].
    # Why it is used: Separates valid records from invalid outlier records.
    # How it works: Splices DataFrame based on boolean mask.
    valid_df = df[valid_mask].copy().reset_index(drop=True)
    invalid_df = df[~valid_mask].copy().reset_index(drop=True)

    return valid_df, invalid_df


if __name__ == "__main__":
    # What is used: Dictionary containing valid and invalid student records.
    # Why it is used: Serves as test data for domain boundary validation.
    # How it works: Holds negative age (-5), age > 100 (150), and invalid marks (110, -20).
    sample = {
        "Name": ["Aarav", "Kabir", "Rohan", "Priya", "Vikram"],
        "Age": [20, -5, 150, 22, 19],
        "Marks": [85, 90, 75, 110, -20]
    }
    df = pd.DataFrame(sample)

    # What is used: Calling validate_student_records.
    # Why it is used: Separates valid records from invalid outliers.
    # How it works: Prints valid and invalid record tables.
    valid_df, invalid_df = validate_student_records(df)

    print("--- Valid Student Records ---")
    print(valid_df)

    print("\n--- Invalid / Out-of-Bounds Records ---")
    print(invalid_df)
