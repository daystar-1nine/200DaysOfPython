"""
Unit tests for Cross-Encoder Reranker and BaseReranker interface.
"""
import pytest
from app.retrieval.reranker import (
    BaseReranker,
    CrossEncoderReranker,
    IdentityReranker,
)
from app.index.vector_index import ScoredChunk
from app.chunking.chunker import Chunk
from app.ingestion.metadata import ChunkMetadata


def make_chunk(cid: str, text: str) -> Chunk:
    return Chunk(text=text, metadata=ChunkMetadata(cid, "doc_" + cid, 0, "Title", "cat", 0, len(text), 10, 10))


def test_identity_reranker():
    reranker = IdentityReranker()
    c1 = make_chunk("c1", "text 1")
    c2 = make_chunk("c2", "text 2")
    cands = [ScoredChunk(c1, 0.9), ScoredChunk(c2, 0.8)]
    out = reranker.rerank("query", cands, top_k=1)
    assert len(out) == 1
    assert out[0].chunk.metadata.chunk_id == "c1"


def test_cross_encoder_reranker_weights_normalization():
    reranker = CrossEncoderReranker(coverage_weight=4, ngram_weight=3, proximity_weight=2, prior_weight=1)
    total = reranker.w_coverage + reranker.w_ngram + reranker.w_proximity + reranker.w_prior
    assert total == pytest.approx(1.0)


def test_cross_encoder_exact_ngram_match_bonus():
    reranker = CrossEncoderReranker()
    
    # Doc 1 has exact consecutive bigram "HTTP 404 error"
    c_exact = make_chunk("c_exact", "An HTTP 404 error is returned by web servers when missing.")
    # Doc 2 has words scattered far apart
    c_scattered = make_chunk("c_scattered", "The error occurred in an HTTP module at line 404.")
    
    score_exact = reranker.score_pair("HTTP 404 error", c_exact)
    score_scattered = reranker.score_pair("HTTP 404 error", c_scattered)
    assert score_exact > score_scattered


def test_cross_encoder_proximity_scoring():
    reranker = CrossEncoderReranker()
    
    # Close proximity
    c_close = make_chunk("c_close", "SQL INNER JOIN matching keys between tables.")
    # Far proximity (50 filler words between terms)
    filler = " " .join(["word"] * 40)
    c_far = make_chunk("c_far", f"SQL {filler} INNER {filler} JOIN tables.")
    
    score_close = reranker.score_pair("SQL INNER JOIN", c_close)
    score_far = reranker.score_pair("SQL INNER JOIN", c_far)
    assert score_close > score_far


def test_cross_encoder_rerank_ordering():
    reranker = CrossEncoderReranker()
    
    # Initial candidates in reverse order: candidate 1 is weak, candidate 2 is high quality
    c_weak = make_chunk("c_weak", "This text talks vaguely about Python programming loops.")
    c_strong = make_chunk("c_strong", "Gradient descent minimizes cost function scaling updates by learning rate.")
    
    initial = [
        ScoredChunk(c_weak, 0.9),
        ScoredChunk(c_strong, 0.5)
    ]
    
    reranked = reranker.rerank("gradient descent learning rate", initial, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].chunk.metadata.chunk_id == "c_strong"
    assert reranked[0].score > reranked[1].score


def test_cross_encoder_empty_query():
    reranker = CrossEncoderReranker()
    c = make_chunk("c", "some text")
    cands = [ScoredChunk(c, 0.5)]
    assert reranker.rerank("", cands) == cands
    assert reranker.rerank("   ", cands) == cands


def test_cross_encoder_empty_candidates():
    reranker = CrossEncoderReranker()
    assert reranker.rerank("query", []) == []


def test_cross_encoder_score_pair_bounds():
    reranker = CrossEncoderReranker()
    c = make_chunk("c", "A document about Python programming.")
    score = reranker.score_pair("python", c)
    assert 0.0 <= score <= 1.0
