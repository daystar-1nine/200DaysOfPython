"""
Unit tests for HybridSearchEngine.
"""

import numpy as np
import pytest
from app.retrieval.tfidf_search import TfidfSearchEngine
from app.retrieval.embedding_search import EmbeddingSearchEngine
from app.retrieval.hybrid_search import HybridSearchEngine
from app.representations.document_embedding import DocumentEmbedder


@pytest.fixture
def indexed_engines():
    corpus = [
        {"id": 1, "title": "Python Language", "text": "Python programming generators decorators", "category": "Python"},
        {"id": 2, "title": "Neural Learning", "text": "Deep neural network gradient backpropagation", "category": "DL"},
        {"id": 3, "title": "Car Service", "text": "Vehicle maintenance auto repair diagnostics", "category": "Auto"}
    ]
    t_engine = TfidfSearchEngine()
    t_engine.index(corpus)

    embedder = DocumentEmbedder(embedding_dim=16, seed=42)
    e_engine = EmbeddingSearchEngine(embedder=embedder)
    e_engine.index(corpus)

    return t_engine, e_engine


def test_hybrid_search_basic(indexed_engines):
    t_eng, e_eng = indexed_engines
    hybrid = HybridSearchEngine(t_eng, e_eng, default_alpha=0.5)

    results = hybrid.search("python", top_k=2)
    assert len(results) == 2
    assert results[0]["document_id"] == 1
    assert "hybrid_score" in results[0]
    assert "tfidf_score" in results[0]
    assert "embedding_score" in results[0]


def test_hybrid_search_pure_tfidf(indexed_engines):
    t_eng, e_eng = indexed_engines
    hybrid = HybridSearchEngine(t_eng, e_eng)

    scores_pure_t = hybrid.get_scores("python", alpha=1.0)
    scores_t = t_eng.get_scores("python")
    # Rank order of top 1 document should match
    assert np.argmax(scores_pure_t) == np.argmax(scores_t)


def test_hybrid_search_pure_embedding(indexed_engines):
    t_eng, e_eng = indexed_engines
    hybrid = HybridSearchEngine(t_eng, e_eng)

    scores_pure_e = hybrid.get_scores("neural", alpha=0.0)
    scores_e = e_eng.get_scores("neural")
    assert np.argmax(scores_pure_e) == np.argmax(scores_e)


def test_hybrid_search_alpha_clipping(indexed_engines):
    t_eng, e_eng = indexed_engines
    hybrid = HybridSearchEngine(t_eng, e_eng)

    scores_neg = hybrid.get_scores("python", alpha=-0.5)
    scores_zero = hybrid.get_scores("python", alpha=0.0)
    assert np.allclose(scores_neg, scores_zero)

    scores_large = hybrid.get_scores("python", alpha=1.5)
    scores_one = hybrid.get_scores("python", alpha=1.0)
    assert np.allclose(scores_large, scores_one)
