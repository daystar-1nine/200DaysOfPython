"""
Unit tests for ContextCompressor and CompressedContext.
"""
import pytest
from app.context.compressor import ContextCompressor, CompressedContext
from app.index.vector_index import ScoredChunk
from app.chunking.chunker import Chunk
from app.ingestion.metadata import ChunkMetadata


def make_chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            chunk_id=cid,
            document_id="doc_" + cid,
            chunk_index=0,
            title="Doc Title",
            category="python",
            start_char=0,
            end_char=len(text),
            token_count=len(text.split()),
            word_count=len(text.split())
        )
    )


def test_context_compressor_empty_chunks():
    compressor = ContextCompressor()
    res = compressor.compress("query", [])
    assert res.text == ""
    assert res.original_char_count == 0
    assert res.compressed_char_count == 0
    assert res.compression_ratio == 0.0


def test_context_compressor_prunes_irrelevant_sentences():
    text = (
        "Python was designed in 1991 by Guido van Rossum. "
        "It features dynamic typing and garbage collection. "
        "The Python Global Interpreter Lock prevents multiple native threads from executing bytecode concurrently. "
        "It is very widely used today for scripting and web backends."
    )
    chunk = make_chunk("c1", text)
    compressor = ContextCompressor(min_sentence_score=0.25)
    
    res = compressor.compress("Global Interpreter Lock GIL threads", [ScoredChunk(chunk, 1.0)])
    assert "Global Interpreter Lock" in res.text
    assert "1991" not in res.text
    assert res.compression_ratio > 0.3
    assert res.retained_sentences_count < res.total_sentences_count


def test_context_compressor_budget_limit():
    # Long text with 10 sentences
    sentences = [f"Sentence number {i} discussing machine learning optimization and gradient descent." for i in range(10)]
    long_text = " ".join(sentences)
    chunk = make_chunk("c1", long_text)
    
    # Restrict budget to 250 characters
    compressor = ContextCompressor(max_context_length=250)
    res = compressor.compress("gradient descent optimization", [ScoredChunk(chunk, 1.0)])
    assert len(res.text) <= 280  # Header + truncated text


def test_context_compressor_preserves_metadata_headers():
    chunk = make_chunk("c1", "Gradient descent updates parameters scaling by learning rate.")
    compressor = ContextCompressor()
    res = compressor.compress("learning rate", [ScoredChunk(chunk, 1.0)])
    assert "[Source: doc_c1 | Chunk: 0 | Category: python]" in res.text


def test_context_compressor_fallback_on_weak_matches():
    # When no sentence strictly exceeds threshold, best matching sentence is retained
    chunk = make_chunk("c1", "Sentence one with very mild match python. Sentence two about completely unrelated cooking.")
    compressor = ContextCompressor(min_sentence_score=0.9)  # very high threshold
    res = compressor.compress("python programming language", [ScoredChunk(chunk, 0.5)])
    assert "Sentence one" in res.text
    assert "cooking" not in res.text
