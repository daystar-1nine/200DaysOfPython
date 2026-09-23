"""
Data validation module for Day 105: Neural NLP & Text Classification.
"""

from typing import List, Tuple
import pandas as pd


class DataValidator:
    """Validates text data integrity, missing values, and class distribution."""

    @staticmethod
    def validate(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate dataset structure, completeness, and balance.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []
        if df.empty:
            return False, ["Dataset is empty."]

        if "text" not in df.columns or "label" not in df.columns:
            return False, ["Dataset must contain 'text' and 'label' columns."]

        # Check for missing values
        null_texts = df["text"].isnull().sum()
        if null_texts > 0:
            errors.append(f"Found {null_texts} null text entries.")

        # Check for empty strings
        empty_texts = (df["text"].str.strip() == "").sum()
        if empty_texts > 0:
            errors.append(f"Found {empty_texts} empty string text entries.")

        # Check valid labels
        valid_labels = {"ham", "spam"}
        found_labels = set(df["label"].unique())
        if not found_labels.issubset(valid_labels):
            errors.append(f"Found invalid labels: {found_labels - valid_labels}")

        return len(errors) == 0, errors

    @staticmethod
    def get_summary(df: pd.DataFrame) -> dict:
        """Return dataset statistics summary."""
        counts = df["label"].value_counts().to_dict()
        total = len(df)
        lengths = df["text"].str.len()
        words = df["text"].apply(lambda s: len(str(s).split()))

        return {
            "total_samples": total,
            "class_counts": counts,
            "class_proportions": {k: v / total for k, v in counts.items()},
            "mean_char_length": float(lengths.mean()),
            "median_char_length": float(lengths.median()),
            "mean_word_count": float(words.mean()),
            "median_word_count": float(words.median()),
            "max_word_count": int(words.max()),
            "min_word_count": int(words.min())
        }
