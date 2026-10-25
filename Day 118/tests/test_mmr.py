"""
Unit tests for Maximal Marginal Relevance (MMR) diversification.
"""
import pytest
import numpy as np
from app.retrieval.mmr import (
    maximal_marginal_relevance,
    MaximalMarginalRelevance,
    jaccard_similarity,
)
from app.index.vector_index import ScoredChunk
from app.chunking.chunker import Chunk
from app.ingestion.metadata import ChunkMetadata


def make_chunk(cid: str, text: str) -> Chunk:
    return Chunk(text=text, metadata=ChunkMetadata(cid, "d1", 0, "T", "cat", 0, len(text), 5, 5))


def test_jaccard_similarity():
    s1 = "python list comprehension memory"
    s2 = "python list comprehension memory"
    s3 = "completely different words here"
    assert jaccard_similarity(s1, s2) == pytest.approx(1.0)
    assert jaccard_similarity(s1, s3) == 0.0
    assert 0.0 < jaccard_similarity(s1, "python list functions") < 1.0


def test_mmr_invalid_lambda():
    with pytest.raises(ValueError):
        maximal_marginal_relevance(None, [], lambda_mult=-0.1)
    with pytest.raises(ValueError):
        maximal_marginal_relevance(None, [], lambda_mult=1.1)


def test_mmr_empty_candidates():
    assert maximal_marginal_relevance(None, []) == []


def test_mmr_top_k_larger_than_candidates():
    c1 = make_chunk("c1", "text 1")
    cands = [ScoredChunk(c1, 1.0)]
    assert maximal_marginal_relevance(None, cands, top_k=5) == cands


def test_mmr_suppresses_redundant_duplicates():
    # Candidates: c1 and c2 are identical; c3 is different
    c1 = make_chunk("c1", "python list comprehension syntax eager memory")
    c2 = make_chunk("c2", "python list comprehension syntax eager memory duplicate")
    c3 = make_chunk("c3", "gradient descent learning rate optimization neural")

    cands = [
        ScoredChunk(c1, 0.95),
        ScoredChunk(c2, 0.94),
        ScoredChunk(c3, 0.70)
    ]

    # lambda = 1.0 (pure relevance: picks c1, c2)
    greedy = maximal_marginal_relevance(None, cands, top_k=2, lambda_mult=1.0)
    assert greedy[0].chunk.metadata.chunk_id == "c1"
    assert greedy[1].chunk.metadata.chunk_id == "c2"

    # lambda = 0.5 (balanced MMR: picks c1, then diverse c3 instead of duplicate c2)
    diverse = maximal_marginal_relevance(None, cands, top_k=2, lambda_mult=0.5)
    assert diverse[0].chunk.metadata.chunk_id == "c1"
    assert diverse[1].chunk.metadata.chunk_id == "c3"


def test_mmr_with_vector_embeddings():
    q_vec = np.array([0.8, 0.0, 0.6])
    c1 = make_chunk("c1", "a")
    c2 = make_chunk("c2", "b")
    c3 = make_chunk("c3", "c")

    embs = np.array([
        [1.0, 0.0, 0.0],   # c1: highest query match (sim=0.80)
        [0.99, 0.01, 0.0], # c2: near duplicate of c1 (sim to c1 = 0.9999)
        [0.1, 0.0, 0.99]   # c3: novel (sim to c1 = 0.10) with strong query match (sim=0.674)
    ])
    cands = [ScoredChunk(c1, 0.80), ScoredChunk(c2, 0.79), ScoredChunk(c3, 0.67)]

    res = maximal_marginal_relevance(
        query_vector=q_vec,
        candidates=cands,
        candidate_embeddings=embs,
        top_k=2,
        lambda_mult=0.5
    )
    assert len(res) == 2
    assert res[0].chunk.metadata.chunk_id == "c1"
    assert res[1].chunk.metadata.chunk_id == "c3"


def test_maximal_marginal_relevance_class_wrapper():
    mmr = MaximalMarginalRelevance(lambda_mult=0.7)
    c1 = make_chunk("c1", "test 1")
    cands = [ScoredChunk(c1, 0.8)]
    selected = mmr.select(cands, top_k=1)
    assert len(selected) == 1
