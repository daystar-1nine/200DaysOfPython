"""Tests for text normalization and data cleaning."""
import pytest
import pandas as pd
from app.preprocessing.normalization import normalize_text
from app.data.cleaner import clean_dataset

def test_normalize_text_lowercase():
    assert normalize_text("HELLO WORLD") == "hello world"

def test_normalize_text_whitespace():
    assert normalize_text("  hello   world  ") == "hello world"

def test_normalize_text_punctuation():
    assert normalize_text("Hello, world! How's it going?") == "hello world how s it going"

def test_normalize_text_url_removal():
    assert normalize_text("Check this out http://example.com/page now") == "check this out now"

def test_normalize_text_email_removal():
    assert normalize_text("Contact me at user@domain.com today") == "contact me at today"

def test_normalize_text_empty_and_none():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""

def test_normalize_text_non_string():
    assert normalize_text(12345) == "12345"

def test_clean_dataset_removes_nulls_and_empty():
    df = pd.DataFrame({
        "label": ["ham", "spam", None, "ham"],
        "text": ["Hello", "   ", "Test", None]
    })
    cleaned = clean_dataset(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["text"] == "Hello"
    assert "char_length" in cleaned.columns
    assert "word_count" in cleaned.columns
