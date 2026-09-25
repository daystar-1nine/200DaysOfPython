"""
Tests for DataCleaner and text normalization in Day 106.
"""

import pandas as pd
from app.data.cleaner import DataCleaner


def test_clean_text_lowercasing():
    raw = "URGENT! Claim Your PRIZE Now"
    cleaned = DataCleaner.clean_text(raw)
    assert cleaned == "urgent! claim your prize now"


def test_clean_text_html_entities():
    raw = "You &amp; Me &lt;3"
    cleaned = DataCleaner.clean_text(raw)
    assert cleaned == "you & me <3"


def test_clean_text_whitespace_normalization():
    raw = "   Too   many     spaces \n\n and tabs \t here   "
    cleaned = DataCleaner.clean_text(raw)
    assert cleaned == "too many spaces and tabs here"


def test_clean_text_empty_and_non_string():
    assert DataCleaner.clean_text("") == ""
    assert DataCleaner.clean_text(None) == ""
    assert DataCleaner.clean_text(12345) == ""


def test_clean_dataframe_pipeline():
    df = pd.DataFrame({
        "label": ["ham", "spam"],
        "text": ["  HELLO WORLD  ", "WIN $500 &amp; CASH!  "]
    })
    cleaned_df = DataCleaner.clean_dataframe(df)
    assert len(cleaned_df) == 2
    assert cleaned_df["text"].iloc[0] == "hello world"
    assert cleaned_df["text"].iloc[1] == "win $500 & cash!"
