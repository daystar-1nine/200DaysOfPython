"""
Module: analyzer.py
Computes data quality metrics, missing value statistics, before/after comparisons, and demographic statistics.
"""

# What is used: Import pandas and numpy modules.
# Why it is used: Core libraries for statistical aggregations and data quality analysis.
# How it works: Brings pandas and numpy into execution context.
import numpy as np
import pandas as pd


def analyze_data_quality(raw_df: pd.DataFrame, clean_df: pd.DataFrame, stats: dict) -> dict:
    """
    Generate comprehensive Data Quality analysis comparing raw vs cleaned DataFrames.

    Args:
        raw_df: Original messy raw DataFrame.
        clean_df: Processed clean DataFrame.
        stats: Audit statistics dictionary from cleaner.

    Returns:
        dict: Analytical quality metrics dictionary.
    """
    analysis = {}

    # 1. Missing Value Statistics (Before vs After)
    raw_missing_count = raw_df.isna().sum().to_dict()
    raw_missing_pct = (raw_df.isna().mean() * 100.0).round(2).to_dict()

    clean_missing_count = clean_df.isna().sum().to_dict()
    clean_missing_pct = (clean_df.isna().mean() * 100.0).round(2).to_dict()

    analysis["missing"] = {
        "raw_count": raw_missing_count,
        "raw_pct": raw_missing_pct,
        "clean_count": clean_missing_count,
        "clean_pct": clean_missing_pct
    }

    # 2. Duplicate Row Statistics
    raw_duplicates = int(raw_df.duplicated().sum())
    clean_duplicates = int(clean_df.duplicated().sum())

    analysis["duplicates"] = {
        "raw_duplicates": raw_duplicates,
        "clean_duplicates": clean_duplicates
    }

    # 3. Clean Dataset Summary Statistics
    # What is used: df.describe() for numerical statistics.
    # Why it is used: Provides count, mean, std, min, median, max for age and salary.
    # How it works: Calculates summary metrics on clean_df.
    num_stats = {}
    for col in ["age", "salary"]:
        if col in clean_df.columns:
            series = clean_df[col]
            num_stats[col] = {
                "count": len(series),
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2),
                "min": round(float(series.min()), 2),
                "median": round(float(series.median()), 2),
                "max": round(float(series.max()), 2)
            }

    analysis["descriptive"] = num_stats

    # 4. Categorical Demographic Distribution
    gender_dist = clean_df["gender"].value_counts().to_dict() if "gender" in clean_df.columns else {}
    dept_dist = clean_df["department"].value_counts().to_dict() if "department" in clean_df.columns else {}
    city_dist = clean_df["city"].value_counts().to_dict() if "city" in clean_df.columns else {}

    analysis["demographics"] = {
        "gender": gender_dist,
        "department": dept_dist,
        "city": city_dist
    }

    return analysis
