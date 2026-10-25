"""
Unit tests for Query Expansion, Query Rewriting, and Multi-Query Generation.
"""
import pytest
from app.retrieval.query_expansion import (
    expand_query,
    rewrite_query,
    generate_subqueries,
    QueryOptimizer,
    TECHNICAL_SYNONYMS,
)
from app.index.vector_index import ScoredChunk
from app.chunking.chunker import Chunk
from app.ingestion.metadata import ChunkMetadata


def test_expand_query_empty():
    assert expand_query("") == ""
    assert expand_query("   ") == ""


def test_expand_query_technical_terms():
    expanded = expand_query("How to fix overfitting?")
    assert "overfitting" in expanded
    assert any(syn in expanded for syn in ["generalization", "regularization", "dropout"])


def test_expand_query_multiple_technical_terms():
    expanded = expand_query("inner join with gradient descent")
    assert "inner join" in expanded
    assert "sql join" in expanded or "relational" in expanded
    assert "learning rate" in expanded or "optimization" in expanded


def test_expand_query_no_synonyms_leaves_query_unchanged():
    q = "random non technical sentence"
    assert expand_query(q) == q


def test_rewrite_conversational_query_prefixes():
    q1 = "Can you please tell me how to reduce overfitting?"
    rw1 = rewrite_query(q1)
    assert not rw1.lower().startswith("can you please")
    assert "overfitting" in rw1

    q2 = "Could you please explain what is an inner join?"
    rw2 = rewrite_query(q2)
    assert not rw2.lower().startswith("could you please")
    assert "inner join" in rw2

    q3 = "What is the meaning of gradient descent?"
    rw3 = rewrite_query(q3)
    assert "gradient descent" in rw3


def test_rewrite_query_strips_punctuation():
    q = "What is an HTTP 404 error???!!"
    rw = rewrite_query(q)
    assert not rw.endswith("?")
    assert not rw.endswith("!")
    assert "http 404" in rw.lower()


def test_generate_subqueries_count():
    subs = generate_subqueries("How to avoid memory leak in Python?", count=3)
    assert len(subs) == 3
    assert all(isinstance(s, str) and len(s) > 0 for s in subs)


def test_generate_subqueries_diversity():
    subs = generate_subqueries("overfitting in neural networks", count=3)
    assert len(set(subs)) == len(subs)  # Deduplicated


def test_query_optimizer_process_query():
    opt = QueryOptimizer(enable_expansion=True, enable_rewriting=True)
    res = opt.process_query("Can you tell me how to prevent overfitting?")
    assert not res.lower().startswith("can you")
    assert "overfitting" in res
    assert any(w in res for w in ["generalization", "regularization", "dropout"])


def test_query_optimizer_multi_query_search():
    opt = QueryOptimizer(enable_multi_query=True, multi_query_count=3)
    
    # Mock search function
    c1 = Chunk(text="t1", metadata=ChunkMetadata("c1", "d1", 0, "T", "cat", 0, 2, 2, 2))
    c2 = Chunk(text="t2", metadata=ChunkMetadata("c2", "d2", 0, "T", "cat", 0, 2, 2, 2))
    
    def mock_search(q: str, k: int):
        if "principles" in q:
            return [ScoredChunk(c1, 1.0)]
        return [ScoredChunk(c2, 1.0)]

    fused = opt.multi_query_search(mock_search, "test query", top_k=2)
    assert len(fused) <= 2
    assert len(fused) > 0
