"""
Unit tests for exact match, normalized exact match, and text canonicalization.
"""
import pytest
from app.evaluators.exact_match import (
    normalize_text,
    exact_match_score,
    normalized_exact_match,
    batch_exact_match
)


def test_normalize_text_lowercasing():
    assert normalize_text("Python ProGraMMing") == "python programming"


def test_normalize_text_punctuation_removal():
    assert normalize_text("Hello, World! How's it going?") == "hello world hows it going"


def test_normalize_text_article_removal():
    assert normalize_text("The capital is a city and an urban hub") == "capital is city and urban hub"


def test_normalize_text_whitespace_collapse():
    assert normalize_text("   data    science     pipeline   ") == "data science pipeline"


def test_normalize_text_empty_input():
    assert normalize_text("") == ""
    assert normalize_text("   \n\t  ") == ""


def test_exact_match_score_identical_strings():
    assert exact_match_score("Paris", "Paris") == 1.0


def test_exact_match_score_case_sensitivity():
    assert exact_match_score("paris", "Paris") == 0.0


def test_exact_match_score_extra_tokens():
    assert exact_match_score("Paris France", "Paris") == 0.0


def test_exact_match_score_multi_reference():
    refs = ["London", "Paris", "Berlin"]
    assert exact_match_score("Paris", refs) == 1.0
    assert exact_match_score("Tokyo", refs) == 0.0


def test_exact_match_score_empty_strings():
    assert exact_match_score("", "") == 1.0
    assert exact_match_score("", "Not empty") == 0.0


def test_normalized_exact_match_punctuation_insensitivity():
    assert normalized_exact_match("Paris.", "Paris") == 1.0
    assert normalized_exact_match("Hello, world!", "hello world") == 1.0


def test_normalized_exact_match_article_insensitivity():
    assert normalized_exact_match("The capital", "Capital") == 1.0
    assert normalized_exact_match("an apple", "apple") == 1.0


def test_normalized_exact_match_casing_insensitivity():
    assert normalized_exact_match("NEW YORK", "new york") == 1.0


def test_normalized_exact_match_disparate_text():
    assert normalized_exact_match("Python 3.12", "Java 17") == 0.0


def test_normalized_exact_match_multi_reference():
    refs = ["The dog", "A canine", "A puppy"]
    assert normalized_exact_match("dog", refs) == 1.0
    assert normalized_exact_match("canine", refs) == 1.0
    assert normalized_exact_match("feline", refs) == 0.0


def test_batch_exact_match_aggregate():
    preds = ["Paris", "Berlin", "tokyo"]
    refs = ["Paris", "Berlin", "Tokyo"]
    res = batch_exact_match(preds, refs)
    assert res["strict_em_rate"] == pytest.approx(2 / 3, rel=1e-2)
    assert res["normalized_em_rate"] == 1.0
