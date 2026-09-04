"""
Day 58 - Coding Challenge 4: Categorical Value Standardization
Standardize messy, inconsistent gender text strings (M, male, MALE, F, female, FEMALE) to Male / Female.
"""

# What is used: Import pandas library.
# Why it is used: Core package for categorical normalization and dictionary replacement.
# How it works: Brings pandas namespace into scope.
import pandas as pd


def standardize_gender_column(series: pd.Series) -> pd.Series:
    """
    Standardize messy Gender string series to 'Male', 'Female', or 'Unknown'.

    Args:
        series: Input Series containing inconsistent gender string representations.

    Returns:
        pd.Series: Standardized Series.
    """
    # What is used: Series.astype(str).str.strip().str.lower().
    # Why it is used: Normalizes text casing and eliminates whitespace.
    # How it works: Converts strings to lowercase.
    clean_str = series.astype(str).str.strip().str.lower()

    # What is used: Dictionary mapping variations to canonical values.
    # Why it is used: Maps inconsistent text codes to standardized labels.
    # How it works: Replaces 'm', 'male', 'males' with 'Male', 'f', 'female' with 'Female'.
    mapping = {
        "m": "Male",
        "male": "Male",
        "males": "Male",
        "f": "Female",
        "female": "Female",
        "females": "Female"
    }

    # What is used: clean_str.map(mapping).fillna("Unknown").
    # Why it is used: Replaces variations and marks unmapped values as 'Unknown'.
    # How it works: Returns standardized Series.
    standardized = clean_str.map(mapping).fillna("Unknown")
    return standardized


if __name__ == "__main__":
    # What is used: Series with messy gender inputs.
    # Why it is used: Test data for categorical standardization.
    # How it works: Contains M, male, MALE, F, female, FEMALE, unknown, nan.
    raw_genders = pd.Series(["M", "male", "MALE ", "F", " female", "FEMALE", "other", None])

    # What is used: Calling standardize_gender_column.
    # Why it is used: Normalizes messy gender inputs to Male / Female / Unknown.
    # How it works: Prints raw vs standardized output.
    clean_genders = standardize_gender_column(raw_genders)
    comparison = pd.DataFrame({"Raw": raw_genders, "Standardized": clean_genders})

    print("--- Gender Categorical Standardization ---")
    print(comparison)
