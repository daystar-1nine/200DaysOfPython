"""Tests for normalization and cleaning strategies."""
import pytest
from app.preprocessing.normalization import normalize_text
from app.preprocessing.cleaner import clean_minimal, clean_punctuation, clean_aggressive

def test_normalize_empty_and_none():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""

def test_normalize_non_string():
    assert normalize_text(999) == "999"

def test_normalize_whitespace_and_casing():
    assert normalize_text("  HELLO   WORLD  ") == "hello world"

def test_normalize_punctuation_removal():
    assert normalize_text("Hello, world! How's it?") == "hello world how s it"

def test_normalize_urls_and_emails():
    assert normalize_text("Check http://test.com or mail me@test.com") == "check or mail"

def test_clean_minimal_keeps_punctuation():
    text = "Hello, World! 123."
    res = clean_minimal(text)
    assert "," in res and "." in res
    assert "hello" in res

def test_clean_punctuation_strips_symbols():
    text = "Win $1000 cash! Call now."
    res = clean_punctuation(text)
    assert "!" not in res and "$" not in res and "." not in res

def test_clean_aggressive_removes_short_and_numbers():
    text = "Win 1000 cash in a box"
    res = clean_aggressive(text)
    assert "1000" not in res
    assert "in" not in res.split()
    assert "win" in res and "cash" in res
