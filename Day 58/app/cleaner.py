"""
Module: cleaner.py
Performs end-to-end data cleaning pipeline, column normalization, type coercion, deduplication, and imputation.
"""

# What is used: Import pandas and numpy modules.
# Why it is used: Core libraries for DataFrame string manipulation, statistical imputation, and type conversions.
# How it works: Brings pandas and numpy into execution context.
import numpy as np
import pandas as pd


def clean_customer_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Execute full data cleaning pipeline on messy raw customer DataFrame.

    Pipeline Steps:
    1. Standardize column names (lowercase, stripped, underscores).
    2. Clean string columns (trim whitespace, title case for names/city/department, lowercase for email).
    3. Normalize phone numbers (strip non-digit characters).
    4. Standardize Gender category (Male / Female / Unknown).
    5. Clean & parse Salary currency strings to numeric float.
    6. Coerce Age to numeric float and set out-of-bounds ages (< 0 or > 120) to NaN.
    7. Coerce Join_Date to datetime.
    8. Deduplicate rows based on customer_id key.
    9. Impute missing values (Age/Salary median, City/Department mode, Email/Phone unknown).
    10. Extract derived features (join_year, join_month, join_month_name).

    Args:
        df: Raw messy customer DataFrame.

    Returns:
        tuple[pd.DataFrame, dict]: Cleaned DataFrame and audit statistics dictionary.
    """
    cleaned_df = df.copy()
    stats = {
        "initial_rows": len(cleaned_df),
        "duplicates_removed": 0,
        "nulls_filled": 0,
        "invalid_ages_corrected": 0,
        "invalid_salaries_corrected": 0,
        "invalid_dates_corrected": 0,
        "final_rows": 0
    }

    # 1. Clean Column Headers
    # What is used: String manipulations on df.columns.
    # Why it is used: Normalizes column names into clean Python identifiers.
    # How it works: Lowercases and replaces spaces with underscores.
    cleaned_df.columns = (
        cleaned_df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # 2. Clean Text Strings (name, city, department, email)
    # What is used: Vectorized string accessors .str.strip(), .str.title(), .str.lower().
    # Why it is used: Removes accidental spaces and normalizes text casing.
    # How it works: Title cases names/city/department, lowercases email.
    for col in ["customer_id", "name", "city", "department"]:
        if col in cleaned_df.columns:
            cleaned_df[col] = (
                cleaned_df[col]
                .astype(str)
                .str.strip()
                .str.title()
                .replace("Nan", np.nan)
                .replace("None", np.nan)
            )

    if "email" in cleaned_df.columns:
        cleaned_df["email"] = (
            cleaned_df["email"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace("nan", np.nan)
            .replace("none", np.nan)
        )

    # 3. Clean Phone Numbers (strip non-digit characters)
    # What is used: Regex replacement .str.replace(r'\D', '', regex=True).
    # Why it is used: Standardizes phone string to digit-only series.
    # How it works: Removes spaces, hyphens, and country code prefixes.
    if "phone" in cleaned_df.columns:
        clean_p = cleaned_df["phone"].astype(str).str.replace(r"\D", "", regex=True)
        cleaned_df["phone"] = clean_p.replace("", np.nan)

    # 4. Standardize Gender Category
    # What is used: Dictionary mapping and lower-casing.
    # Why it is used: Normalizes M, male, MALE, F, female, FEMALE to 'Male' or 'Female'.
    # How it works: Maps text variations to canonical gender labels.
    if "gender" in cleaned_df.columns:
        g_clean = cleaned_df["gender"].astype(str).str.strip().str.lower()
        g_map = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
        cleaned_df["gender"] = g_clean.map(g_map).fillna("Unknown")

    # 5. Clean & Parse Salary Monetary String
    # What is used: String symbol replacement and pd.to_numeric(errors="coerce").
    # Why it is used: Parses currency strings (₹60,000, -50000, unknown) into float numbers.
    # How it works: Replaces ₹, $, commas and converts unparseable strings to NaNs.
    if "salary" in cleaned_df.columns:
        sal_str = (
            cleaned_df["salary"]
            .astype(str)
            .str.replace("₹", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        cleaned_df["salary"] = pd.to_numeric(sal_str, errors="coerce")
        # Flag negative salaries as invalid and set to NaN
        neg_sal_mask = cleaned_df["salary"] < 0
        stats["invalid_salaries_corrected"] += int(neg_sal_mask.sum())
        cleaned_df.loc[neg_sal_mask, "salary"] = np.nan

    # 6. Clean & Coerce Age to Numeric Float & Validate Range (0-120)
    # What is used: pd.to_numeric(errors="coerce") and domain boundary check.
    # Why it is used: Coerces invalid string ages and flags impossible ages (-5, 150, 200).
    # How it works: Converts strings to floats and sets values outside 0-120 to NaN.
    if "age" in cleaned_df.columns:
        cleaned_df["age"] = pd.to_numeric(cleaned_df["age"], errors="coerce")
        out_bounds_age = ~cleaned_df["age"].between(0, 120) & cleaned_df["age"].notna()
        stats["invalid_ages_corrected"] += int(out_bounds_age.sum())
        cleaned_df.loc[out_bounds_age, "age"] = np.nan

    # 7. Clean & Parse Join_Date Datetime
    # What is used: pd.to_datetime(format="mixed", errors="coerce").
    # Why it is used: Parses mixed date strings into Timestamps, coercing unparseable dates to NaT.
    # How it works: Converts date formats to datetime64[ns] and fills NaTs with default timestamp.
    if "join_date" in cleaned_df.columns:
        cleaned_df["join_date"] = pd.to_datetime(cleaned_df["join_date"], format="mixed", errors="coerce")
        stats["invalid_dates_corrected"] += int(cleaned_df["join_date"].isna().sum())
        default_date = pd.Timestamp("2026-01-01")
        cleaned_df["join_date"] = cleaned_df["join_date"].fillna(default_date)

    # 8. Deduplicate Records on customer_id Key
    # What is used: df.drop_duplicates(subset=["customer_id"], keep="first").
    # Why it is used: Eliminates duplicate customer records.
    # How it works: Retains first occurrence of each customer_id.
    if "customer_id" in cleaned_df.columns:
        pre_dedupe = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(subset=["customer_id"], keep="first")
        stats["duplicates_removed"] = pre_dedupe - len(cleaned_df)

    # 9. Impute Missing Values
    # What is used: Median for Age/Salary; Mode for City/Department; Unknown for text/email/phone.
    # Why it is used: Safely imputes missing slots using domain-appropriate strategies.
    # How it works: Calculates column statistics ignoring NaNs and fills missing slots.
    num_cols = ["age", "salary"]
    stats["nulls_filled"] += int(cleaned_df[num_cols].isna().sum().sum())

    for col in num_cols:
        if col in cleaned_df.columns:
            med = cleaned_df[col].median()
            fill_v = round(float(med), 1) if not pd.isna(med) else (25.0 if col == "age" else 50000.0)
            cleaned_df[col] = cleaned_df[col].fillna(fill_v)

    text_cat_cols = ["city", "department"]
    stats["nulls_filled"] += int(cleaned_df[text_cat_cols].isna().sum().sum())

    for col in text_cat_cols:
        if col in cleaned_df.columns:
            mode_v = cleaned_df[col].mode()
            fill_v = str(mode_v[0]) if len(mode_v) > 0 else "Unknown"
            cleaned_df[col] = cleaned_df[col].fillna(fill_v)

    if "email" in cleaned_df.columns:
        cleaned_df["email"] = cleaned_df["email"].fillna("unknown@example.com")
    if "phone" in cleaned_df.columns:
        cleaned_df["phone"] = cleaned_df["phone"].fillna("Unknown")

    # 10. Extract Derived Date Features
    # What is used: Datetime accessor .dt.year, .dt.month, .dt.month_name().
    # Why it is used: Creates analytical derived features for downstream reporting.
    # How it works: Extracts year and month components from join_date.
    if "join_date" in cleaned_df.columns:
        cleaned_df["join_year"] = cleaned_df["join_date"].dt.year.fillna(2026).astype(int)
        cleaned_df["join_month"] = cleaned_df["join_date"].dt.month.fillna(1).astype(int)
        cleaned_df["join_month_name"] = cleaned_df["join_date"].dt.month_name().fillna("January")

    # Reset Index Continuously
    cleaned_df = cleaned_df.reset_index(drop=True)
    stats["final_rows"] = len(cleaned_df)

    return cleaned_df, stats
