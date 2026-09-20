"""
Unit tests for DataLoader and DataValidator.
"""

import json
from pathlib import Path
import pytest
from app.data.loader import DataLoader
from app.data.validator import DataValidator
from app.data.ground_truth import GroundTruthManager


def test_load_documents_success(tmp_path):
    docs = [
        {"id": 1, "title": "Doc 1", "text": "Text 1", "category": "Python"},
        {"id": 2, "title": "Doc 2", "text": "Text 2", "category": "ML"}
    ]
    f = tmp_path / "docs.json"
    DataLoader.save_json(docs, f)

    loaded = DataLoader.load_documents(f)
    assert len(loaded) == 2
    assert loaded[0]["id"] == 1


def test_load_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        DataLoader.load_documents("non_existent_file.json")


def test_load_invalid_json_raises_error(tmp_path):
    f = tmp_path / "invalid.json"
    f.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError):
        DataLoader.load_documents(f)


def test_validate_documents_valid():
    docs = [
        {"id": 1, "title": "Doc 1", "text": "Content", "category": "Python"},
        {"id": 2, "title": "Doc 2", "text": "Content", "category": "Cloud"}
    ]
    valid, errors = DataValidator.validate_documents(docs)
    assert valid is True
    assert len(errors) == 0


def test_validate_documents_duplicate_id():
    docs = [
        {"id": 1, "title": "Doc 1", "text": "Content", "category": "Python"},
        {"id": 1, "title": "Doc 2", "text": "Content", "category": "Cloud"}
    ]
    valid, errors = DataValidator.validate_documents(docs)
    assert valid is False
    assert any("Duplicate document id" in e for e in errors)


def test_validate_documents_missing_field():
    docs = [
        {"id": 1, "title": "Doc 1", "category": "Python"}  # missing 'text'
    ]
    valid, errors = DataValidator.validate_documents(docs)
    assert valid is False
    assert any("missing required fields" in e for e in errors)


def test_validate_documents_empty_list():
    valid, errors = DataValidator.validate_documents([])
    assert valid is False


def test_validate_ground_truth_invalid_ref():
    queries = [
        {"query_id": "Q1", "query": "test", "relevant_doc_ids": [999]}
    ]
    valid, errors = DataValidator.validate_ground_truth(queries, valid_doc_ids={1, 2})
    assert valid is False
    assert any("references non-existent doc IDs" in e for e in errors)


def test_ground_truth_manager():
    queries = [
        {"query_id": "Q1", "query": "python generators", "relevant_doc_ids": [1], "category": "Python", "query_type": "keyword"},
        {"query_id": "Q2", "query": "docker", "relevant_doc_ids": [2], "category": "Cloud", "query_type": "short"}
    ]
    gt = GroundTruthManager(queries)
    assert len(gt) == 2
    assert gt.get_relevant_ids("Q1") == [1]
    assert gt.get_relevant_ids("Q99") == []
    assert len(gt.get_queries_by_type("short")) == 1
    assert "Python" in gt.get_categories()
