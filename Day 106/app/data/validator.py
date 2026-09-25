"""
Data validation module for Day 106: RNNs & Sequential Text Learning.
"""

from typing import List, Tuple
import pandas as pd


class DataValidator:
    """Validates text data integrity, missing values, and class balance."""

    @staticmethod
    def validate(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate dataset structure, completeness, and labels.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []
        if df is None or df.empty:
            return False, ["Dataset is empty or None."]

        if "text" not in df.columns or "label" not in df.columns:
            return False, ["Dataset must contain 'text' and 'label' columns."]

        # Check for missing values
        null_texts = df["text"].isnull().sum()
        if null_texts > 0:
            errors.append(f"Found {null_texts} null text entries.")

        null_labels = df["label"].isnull().sum()
        if null_labels > 0:
            errors.append(f"Found {null_labels} null label entries.")

        # Check for empty strings
        empty_texts = (df["text"].astype(str).str.strip() == "").sum()
        if empty_texts > 0:
            errors.append(f"Found {empty_texts} empty string text entries.")

        # Check valid labels
        valid_labels = {"ham", "spam"}
        found_labels = set(df["label"].astype(str).str.lower().unique())
        if not found_labels.issubset(valid_labels):
            errors.append(f"Found invalid labels: {found_labels - valid_labels}")

        return len(errors) == 0, errors
