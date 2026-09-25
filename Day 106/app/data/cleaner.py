"""
Text cleaning and normalization module for Day 106: RNNs & Sequential Text Learning.
"""

import re
import html
import pandas as pd


class DataCleaner:
    """Preprocesses and normalizes raw message text for sequential modeling."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean an individual text string while preserving semantic markers.
        
        Args:
            text: Raw input string.
            
        Returns:
            Normalized string.
        """
        if not isinstance(text, str):
            return ""

        # Unescape HTML entities (e.g., &amp; -> &)
        text = html.unescape(text)

        # Standardize whitespace (newlines, tabs, multiple spaces)
        text = re.sub(r"\s+", " ", text).strip()

        # Lowercase for case-insensitive tokenization
        text = text.lower()

        return text

    @classmethod
    def clean_dataframe(cls, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """Apply text normalization across a DataFrame.
        
        Args:
            df: Input DataFrame.
            text_col: Name of column containing message strings.
            
        Returns:
            DataFrame with normalized text column.
        """
        cleaned_df = df.copy()
        cleaned_df[text_col] = cleaned_df[text_col].apply(cls.clean_text)
        cleaned_df = cleaned_df[cleaned_df[text_col].str.len() > 0].reset_index(drop=True)
        return cleaned_df
