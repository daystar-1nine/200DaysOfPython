"""
Unit tests for search engines: TF-IDF Search and Embedding Search.
"""

import pytest
from app.retrieval.tfidf_search import TfidfSearchEngine
from app.retrieval.embedding_search import EmbeddingSearchEngine
from app.representations.document_embedding import DocumentEmbedder


@pytest.fixture
def sample_corpus():
    return [
        {"id": 1, "title": "Python Basics", "text": "Python programming language with generators and decorators", "category": "Python"},
        {"id": 2, "title": "Deep Learning", "text": "Neural networks training using backpropagation and gradient descent", "category": "Deep Learning"},
        {"id": 3, "title": "Car Repair", "text": "Vehicle maintenance and automotive diagnostics services", "category": "Automotive"}
    ]


def test_tfidf_search_unindexed():
    engine = TfidfSearchEngine()
    with pytest.raises(RuntimeError):
        engine.search("python")


def test_tfidf_search_basic(sample_corpus):
    engine = TfidfSearchEngine()
    engine.index(sample_corpus)
    results = engine.search("python decorators", top_k=2)
    assert len(results) == 2
    assert results[0]["document_id"] == 1
    assert results[0]["rank"] == 1
    assert results[0]["score"] > 0.0


def test_tfidf_search_empty_query(sample_corpus):
    engine = TfidfSearchEngine()
    engine.index(sample_corpus)
    results = engine.search("", top_k=2)
    assert len(results) == 2
    assert results[0]["score"] == 0.0


def test_embedding_search_unindexed():
    engine = EmbeddingSearchEngine()
    with pytest.raises(RuntimeError):
        engine.search("neural")


def test_embedding_search_basic(sample_corpus):
    embedder = DocumentEmbedder(embedding_dim=16, seed=42)
    engine = EmbeddingSearchEngine(embedder=embedder)
    engine.index(sample_corpus)

    results = engine.search("neural networks", top_k=2)
    assert len(results) == 2
    assert results[0]["document_id"] == 2


def test_embedding_search_empty_query(sample_corpus):
    embedder = DocumentEmbedder(embedding_dim=16, seed=42)
    engine = EmbeddingSearchEngine(embedder=embedder)
    engine.index(sample_corpus)

    results = engine.search("", top_k=2)
    assert len(results) == 2
    assert results[0]["score"] == 0.0


def test_search_top_k_larger_than_corpus(sample_corpus):
    engine = TfidfSearchEngine()
    engine.index(sample_corpus)
    results = engine.search("python", top_k=100)
    assert len(results) == len(sample_corpus)
