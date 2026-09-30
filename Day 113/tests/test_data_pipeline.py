"""
Unit tests for Day 113 Data Pipeline:
- Text loading and chunking
- Normalization and quality scoring
- Exact and near deduplication
- Leak-free train/validation splitting
- Data leakage detection and contamination audits
- Sequence packing
"""
import pytest
from app.data.loader import split_into_documents, build_character_vocab
from app.data.cleaner import normalize_text, quality_score, filter_documents
from app.data.dedup import hash_document, exact_deduplicate, get_shingles, jaccard_similarity, near_deduplicate
from app.data.splitter import train_val_split, detect_leakage, inject_leakage, pack_sequences


class TestTextCleanerAndQuality:
    def test_normalize_text_strips_control_characters(self):
        dirty = "Line 1\x00\x07with bell\nLine 2\x1f"
        clean = normalize_text(dirty)
        assert "\x00" not in clean
        assert "\x07" not in clean
        assert "\x1f" not in clean
        assert "Line 1 with bell" in clean

    def test_normalize_text_collapses_whitespace(self):
        spaced = "Hello     world!   This   is   clean."
        assert normalize_text(spaced) == "Hello world! This is clean."

    def test_normalize_text_collapses_excessive_newlines(self):
        newlines = "Para 1\n\n\n\n\nPara 2"
        assert normalize_text(newlines) == "Para 1\n\nPara 2"

    def test_normalize_empty_text(self):
        assert normalize_text("") == ""
        assert normalize_text("   \n\t  ") == ""

    def test_quality_score_empty_or_whitespace_is_zero(self):
        assert quality_score("") == 0.0
        assert quality_score("   ") == 0.0

    def test_quality_score_ultra_short_text(self):
        score = quality_score("Hi")
        assert score <= 0.1

    def test_quality_score_high_quality_text(self):
        doc = "The quick brown fox jumps over the lazy dog. Language models learn broad semantic structures."
        score = quality_score(doc)
        assert score >= 0.85

    def test_quality_score_penalizes_repetitive_characters(self):
        spam = "Check this out aaaaaaaaaaaaaaaaaaaaaaaa please buy now!"
        normal = "Check this out and please purchase this item now!"
        assert quality_score(spam) < quality_score(normal)

    def test_quality_score_penalizes_urls_and_html(self):
        markup = "<p>Click <a href='https://spam.com/123'>here</a> to win $$$</p>"
        assert quality_score(markup) <= 0.65

    def test_filter_documents_partitions_clean_and_rejected(self, sample_documents):
        retained, rejected = filter_documents(sample_documents, min_quality=0.50)
        assert len(retained) == 3
        assert len(rejected) == 2
        assert all("score" in r for r in rejected)


class TestDeduplication:
    def test_hash_document_deterministic(self):
        doc = "Standardization matters for language modeling."
        assert hash_document(doc) == hash_document(doc)
        assert hash_document(doc) == hash_document("  Standardization matters for language modeling.  \n")

    def test_exact_deduplicate_removes_duplicates(self):
        docs = ["Alpha", "Beta", "alpha", "Beta", "Gamma", "Alpha"]
        unique, count = exact_deduplicate(docs)
        # Note: hash_document normalizes case and whitespace
        assert len(unique) == 3
        assert count == 3
        assert unique == ["Alpha", "Beta", "Gamma"]

    def test_exact_deduplicate_empty_list(self):
        unique, count = exact_deduplicate([])
        assert unique == []
        assert count == 0

    def test_get_shingles_short_text(self):
        shingles = get_shingles("hi", n=3)
        assert shingles == {"hi"}

    def test_get_shingles_extraction(self):
        shingles = get_shingles("abcdef", n=3)
        assert "abc" in shingles
        assert "bcd" in shingles
        assert "cde" in shingles
        assert "def" in shingles
        assert len(shingles) == 4

    def test_jaccard_similarity_identical_sets(self):
        s = {"a", "b", "c"}
        assert jaccard_similarity(s, s) == 1.0

    def test_jaccard_similarity_disjoint_sets(self):
        s1 = {"a", "b"}
        s2 = {"c", "d"}
        assert jaccard_similarity(s1, s2) == 0.0

    def test_near_deduplicate_filters_similar_texts(self):
        docs = [
            "The quick brown fox jumps over the lazy dog.",
            "The quick brown fox jumps over the lazy dogs.",  # Near duplicate (plural)
            "A completely unrelated sentence about quantum electrodynamics."
        ]
        unique, dropped = near_deduplicate(docs, similarity_threshold=0.75, shingle_size=3)
        assert len(unique) == 2
        assert dropped == 1
        assert "quantum" in unique[1]


class TestSplitterAndLeakage:
    def test_train_val_split_proportions(self):
        docs = [f"Document {i}" for i in range(100)]
        train, val = train_val_split(docs, val_ratio=0.20, seed=42)
        assert len(train) == 80
        assert len(val) == 20

    def test_train_val_split_empty(self):
        train, val = train_val_split([])
        assert train == []
        assert val == []

    def test_train_val_split_reproducibility(self):
        docs = [f"Doc {i}" for i in range(50)]
        t1, v1 = train_val_split(docs, seed=123)
        t2, v2 = train_val_split(docs, seed=123)
        assert t1 == t2
        assert v1 == v2

    def test_detect_leakage_clean_split(self):
        train = ["Document about planetary science.", "Document about biology."]
        val = ["Sentence discussing economics and inflation."]
        audit = detect_leakage(train, val, ngram_size=5)
        assert audit["exact_leaked_documents"] == 0
        assert audit["has_leakage"] is False

    def test_detect_leakage_exact_match_detected(self):
        train = ["Unique doc A", "Common leaked doc B"]
        val = ["Common leaked doc B", "Unique doc C"]
        audit = detect_leakage(train, val)
        assert audit["exact_leaked_documents"] == 1
        assert audit["has_leakage"] is True

    def test_inject_leakage_modifies_train(self):
        train = ["Doc 1", "Doc 2"]
        val = ["Val 1", "Val 2", "Val 3", "Val 4"]
        contaminated = inject_leakage(train, val, leakage_fraction=0.50)
        assert len(contaminated) == 2 + 2
        assert "Val 1" in contaminated
        assert "Val 2" in contaminated


class TestSequencePacking:
    def test_pack_sequences_exact_fit(self):
        tokens = list(range(32))
        chunks, remainder = pack_sequences(tokens, context_length=16)
        assert len(chunks) == 2
        assert remainder == 0
        assert chunks[0] == list(range(16))
        assert chunks[1] == list(range(16, 32))

    def test_pack_sequences_with_remainder(self):
        tokens = list(range(35))
        chunks, remainder = pack_sequences(tokens, context_length=10)
        assert len(chunks) == 3
        assert remainder == 5

    def test_pack_sequences_too_short_yields_empty(self):
        tokens = [1, 2, 3]
        chunks, remainder = pack_sequences(tokens, context_length=10)
        assert chunks == []
        assert remainder == 3
