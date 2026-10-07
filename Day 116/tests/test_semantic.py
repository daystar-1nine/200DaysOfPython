"""
Unit tests for semantic similarity, vector cosine distance, and BERTScore token alignment.
"""
import pytest
from app.evaluators.semantic import (
    cosine_similarity,
    extract_semantic_features,
    semantic_similarity,
    bertscore_simulation
)


def test_cosine_similarity_identical_vectors():
    vec = {"word1": 1.0, "word2": 2.5, "word3": 0.8}
    assert cosine_similarity(vec, vec) == pytest.approx(1.0, abs=1e-4)


def test_cosine_similarity_disjoint_vectors():
    v1 = {"apple": 1.0, "orange": 2.0}
    v2 = {"computer": 1.0, "keyboard": 3.0}
    assert cosine_similarity(v1, v2) == 0.0


def test_cosine_similarity_empty_vectors():
    assert cosine_similarity({}, {"a": 1.0}) == 0.0
    assert cosine_similarity({}, {}) == 0.0


def test_extract_semantic_features_keys():
    features = extract_semantic_features("Python code")
    assert any(k.startswith("w1_python") for k in features)
    assert any(k.startswith("w1_code") for k in features)
    assert any(k.startswith("w2_python_code") for k in features)


def test_semantic_similarity_identical_sentences():
    text = "Neural networks learn representations through backpropagation."
    sim = semantic_similarity(text, text)
    assert sim == 1.0


def test_semantic_similarity_paraphrase():
    s1 = "Artificial intelligence is transforming society and technology."
    s2 = "AI technology is rapidly reshaping human society."
    sim = semantic_similarity(s1, s2)
    assert sim > 0.15  # Shared roots and subwords


def test_semantic_similarity_disjoint_topics():
    s1 = "Photosynthesis requires sunlight water and carbon dioxide."
    s2 = "Postgres databases use WAL logs for durability."
    sim = semantic_similarity(s1, s2)
    assert sim < 0.15


def test_semantic_similarity_multi_reference():
    pred = "The capital of Germany is Berlin."
    refs = [
        "Tokyo is the capital of Japan.",
        "Berlin is the federal capital of Germany."
    ]
    sim = semantic_similarity(pred, refs)
    assert sim > 0.6


def test_bertscore_simulation_identical_text():
    text = "Gradient boosting trains decision trees sequentially."
    res = bertscore_simulation(text, text)
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0


def test_bertscore_simulation_disjoint_text():
    res = bertscore_simulation("quantum physics", "culinary recipes")
    assert res["f1"] < 0.2


def test_bertscore_simulation_empty_prediction():
    res = bertscore_simulation("", "Some reference sentence")
    assert res["f1"] == 0.0


def test_bertscore_simulation_partial_overlap():
    pred = "Deep learning uses neural networks."
    ref = "Deep learning utilizes deep neural architectures."
    res = bertscore_simulation(pred, ref)
    assert res["precision"] > 0.5
    assert res["recall"] > 0.4
    assert res["f1"] > 0.5
