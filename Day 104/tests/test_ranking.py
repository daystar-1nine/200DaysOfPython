"""
Unit tests for rank_top_k and ranking utilities.
"""

import numpy as np
import pytest
from app.retrieval.ranking import rank_top_k


def test_rank_top_k_sorting():
    scores = np.array([0.1, 0.9, 0.4, 0.7])
    docs = [
        {"id": 1, "title": "Doc 1", "text": "Content 1"},
        {"id": 2, "title": "Doc 2", "text": "Content 2"},
        {"id": 3, "title": "Doc 3", "text": "Content 3"},
        {"id": 4, "title": "Doc 4", "text": "Content 4"}
    ]
    results = rank_top_k(scores, docs, top_k=3)
    assert len(results) == 3
    assert results[0]["document_id"] == 2
    assert results[0]["score"] == 0.9
    assert results[1]["document_id"] == 4
    assert results[1]["score"] == 0.7
    assert results[2]["document_id"] == 3
    assert results[2]["score"] == 0.4


def test_rank_top_k_zero():
    scores = np.array([0.5, 0.2])
    docs = [{"id": 1}, {"id": 2}]
    assert rank_top_k(scores, docs, top_k=0) == []
    assert rank_top_k(scores, docs, top_k=-1) == []


def test_rank_top_k_length_mismatch():
    scores = np.array([0.5])
    docs = [{"id": 1}, {"id": 2}]
    with pytest.raises(ValueError):
        rank_top_k(scores, docs, top_k=1)


def test_rank_top_k_snippet_truncation():
    long_text = "word " * 100
    docs = [{"id": 1, "title": "Long Doc", "text": long_text}]
    results = rank_top_k(np.array([0.8]), docs, top_k=1)
    assert len(results[0]["snippet"]) <= 125
    assert results[0]["snippet"].endswith("...")
