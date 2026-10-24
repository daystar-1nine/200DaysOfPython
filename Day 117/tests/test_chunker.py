"""
Unit tests for text chunking: character, word, and sentence strategies with boundary preservation.
"""
import pytest
from app.chunking.chunker import TextChunker, Chunk, chunk_text
from app.ingestion.metadata import ChunkMetadata
from app.ingestion.loader import Document


def test_chunk_text_function_character():
    text = "0123456789ABCDEFGHIJ"
    chunks = chunk_text(text, chunk_size=10, overlap=2, strategy="character")
    assert len(chunks) == 3
    assert chunks[0].text == "0123456789"
    # stride = 8 -> next chunk starts at index 8
    assert chunks[1].text == "89ABCDEFGH"


def test_chunker_init_validations():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        TextChunker(chunk_size=0)
    with pytest.raises(ValueError, match="overlap cannot be negative"):
        TextChunker(chunk_size=100, overlap=-5)
    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        TextChunker(chunk_size=100, overlap=100)
    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        TextChunker(chunk_size=100, overlap=120)


def test_chunk_empty_text():
    chunker = TextChunker(chunk_size=100, overlap=20)
    assert chunker.chunk_text("") == []
    assert chunker.chunk_text("   \n\t ") == []


def test_chunk_character_strategy():
    chunker = TextChunker(chunk_size=30, overlap=10, strategy="character")
    text = "Machine learning algorithms learn patterns directly from empirical data."
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 30


def test_chunk_word_strategy():
    chunker = TextChunker(chunk_size=6, overlap=2, strategy="word")
    text = "one two three four five six seven eight nine ten eleven twelve"
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 3
    # First chunk has 6 words
    assert len(chunks[0].text.split()) == 6
    assert chunks[0].text == "one two three four five six"
    # Second chunk begins after stride (4 words) -> "five six seven eight nine ten"
    assert "five" in chunks[1].text


def test_chunk_sentence_strategy_preserves_sentences():
    chunker = TextChunker(chunk_size=100, overlap=20, strategy="sentence")
    text = "Sentence Alpha is short. Sentence Beta provides clear details. Sentence Gamma concludes the text."
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 1
    for c in chunks:
        # Each chunk should end with terminal punctuation
        assert c.text.endswith((".", "!", "?"))


def test_chunk_sentence_backtracking_overlap():
    chunker = TextChunker(chunk_size=50, overlap=20, strategy="sentence")
    text = "First small sentence. Second sentence goes here. Third sentence finishes it up."
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 2


def test_chunk_single_long_sentence():
    chunker = TextChunker(chunk_size=40, overlap=10, strategy="sentence")
    long_sent = "This is a ridiculously long single sentence that clearly exceeds the chunk size limit."
    chunks = chunker.chunk_text(long_sent)
    assert len(chunks) > 1


def test_chunk_document_inherits_metadata(sample_documents):
    doc = sample_documents[0]
    chunker = TextChunker(chunk_size=100, overlap=20, strategy="sentence")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 0
    c0 = chunks[0]
    assert c0.metadata.document_id == doc.metadata.document_id
    assert c0.metadata.category == doc.metadata.category
    assert c0.metadata.title == doc.metadata.title
    assert c0.metadata.chunk_id == f"{doc.metadata.document_id}_c0"


def test_chunk_span_offsets():
    text = "Chunk zero string. Chunk one string."
    chunker = TextChunker(chunk_size=25, overlap=5, strategy="character")
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 2
    assert chunks[0].metadata.start_char == 0
    assert chunks[0].metadata.end_char <= 25


def test_chunk_to_dict_and_from_dict():
    meta = ChunkMetadata(
        chunk_id="cid_1",
        document_id="did_1",
        chunk_index=0,
        title="Title",
        category="cat",
        start_char=0,
        end_char=50,
        token_count=10,
        word_count=10,
        source_file="file.md"
    )
    chunk = Chunk(text="Sample chunk content text", metadata=meta)
    d = chunk.to_dict()
    assert d["text"] == "Sample chunk content text"
    assert d["metadata"]["chunk_id"] == "cid_1"

    restored = Chunk.from_dict(d)
    assert restored.text == chunk.text
    assert restored.metadata.chunk_id == "cid_1"
    assert restored.metadata.category == "cat"


def test_chunk_text_auto_scales_overlap():
    # If chunk_size is 20 and default overlap would be 50, it must scale down
    chunks = chunk_text("A quick brown fox jumps over lazy dog.", chunk_size=20)
    assert len(chunks) > 0


def test_chunking_exact_multiple_of_stride():
    # 20 chars, chunk_size=10, overlap=0 -> stride 10 -> exactly 2 chunks
    text = "abcdefghij0123456789"
    chunks = chunk_text(text, chunk_size=10, overlap=0, strategy="character")
    assert len(chunks) == 2
    assert chunks[0].text == "abcdefghij"
    assert chunks[1].text == "0123456789"


def test_chunk_word_single_word_document():
    chunker = TextChunker(chunk_size=10, overlap=2, strategy="word")
    chunks = chunker.chunk_text("Antigravity")
    assert len(chunks) == 1
    assert chunks[0].text == "Antigravity"


def test_chunk_character_single_char():
    chunker = TextChunker(chunk_size=5, overlap=1, strategy="character")
    chunks = chunker.chunk_text("Z")
    assert len(chunks) == 1
    assert chunks[0].text == "Z"


def test_chunk_whitespace_preservation():
    chunker = TextChunker(chunk_size=50, overlap=10, strategy="word")
    text = "word1    word2    word3    word4"
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1
    assert "word1" in chunks[0].text


def test_chunk_strategy_case_insensitive():
    c_lower = TextChunker(chunk_size=50, overlap=10, strategy="sentence")
    c_upper = TextChunker(chunk_size=50, overlap=10, strategy="SENTENCE")
    assert c_lower.strategy == "sentence"
    assert c_upper.strategy == "sentence"


def test_chunk_metadata_word_count_matches():
    chunker = TextChunker(chunk_size=50, overlap=10, strategy="sentence")
    text = "This chunk has exactly seven individual words."
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].metadata.word_count == 7


def test_chunk_consecutive_chunks_overlap_check():
    text = "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa"
    chunker = TextChunker(chunk_size=5, overlap=2, strategy="word")
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 2
    # Ensure overlap words exist in both chunk 0 and chunk 1
    c0_words = set(chunks[0].text.split())
    c1_words = set(chunks[1].text.split())
    assert len(c0_words & c1_words) >= 1


def test_chunk_sentence_punctuation_varieties():
    text = "What is RAG? RAG is retrieval! It works wonders. Indeed."
    chunker = TextChunker(chunk_size=50, overlap=10, strategy="sentence")
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 2


def test_chunk_text_custom_doc_id():
    chunker = TextChunker(chunk_size=40, overlap=5)
    chunks = chunker.chunk_text("Test custom ID generation.", doc_id="custom_doc_99")
    assert chunks[0].metadata.chunk_id == "custom_doc_99_c0"
    assert chunks[0].metadata.document_id == "custom_doc_99"


def test_chunk_document_override_chunk_size(sample_documents):
    doc = sample_documents[0]
    chunker = TextChunker(chunk_size=200, overlap=20)
    # Override with smaller size at call time
    chunks = chunker.chunk_document(doc, chunk_size=50, overlap=10)
    for c in chunks:
        assert len(c.text) <= 60
