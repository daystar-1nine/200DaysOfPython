"""
Unit tests for document representations and pooling strategies.
"""

import numpy as np
import pytest
from app.representations.pooling import mean_pooling, weighted_pooling
from app.representations.document_embedding import DocumentEmbedder


def test_mean_pooling_basic():
    embeddings = {
        "cat": np.array([0.2, 0.4, 0.1]),
        "dog": np.array([0.3, 0.5, 0.2])
    }
    vec = mean_pooling(["cat", "dog"], embeddings, embedding_dim=3)
    assert np.allclose(vec, [0.25, 0.45, 0.15])


def test_mean_pooling_empty_doc():
    embeddings = {"cat": np.array([0.2, 0.4, 0.1])}
    vec = mean_pooling([], embeddings, embedding_dim=3)
    assert np.allclose(vec, np.zeros(3))


def test_mean_pooling_unknown_word_zero():
    embeddings = {"cat": np.array([0.2, 0.4, 0.1])}
    vec = mean_pooling(["cat", "dragon"], embeddings, embedding_dim=3, unk_strategy="zero")
    assert np.allclose(vec, [0.2, 0.4, 0.1])


def test_mean_pooling_all_unknown():
    embeddings = {"cat": np.array([0.2, 0.4, 0.1])}
    vec = mean_pooling(["dragon", "wizard"], embeddings, embedding_dim=3, unk_strategy="zero")
    assert np.allclose(vec, np.zeros(3))


def test_mean_pooling_unk_vector():
    embeddings = {"cat": np.array([0.2, 0.4, 0.1])}
    unk = np.array([0.0, 0.0, 1.0])
    vec = mean_pooling(["cat", "dragon"], embeddings, embedding_dim=3, unk_vector=unk, unk_strategy="unk")
    assert np.allclose(vec, [0.1, 0.2, 0.55])


def test_weighted_pooling_basic():
    embeddings = {
        "cat": np.array([0.2, 0.4, 0.1]),
        "dog": np.array([0.3, 0.5, 0.2])
    }
    weights = {"cat": 2.0, "dog": 1.0}
    vec = weighted_pooling(["cat", "dog"], embeddings, weights=weights, embedding_dim=3)
    expected = (2.0 * embeddings["cat"] + 1.0 * embeddings["dog"]) / 3.0
    assert np.allclose(vec, expected)


def test_weighted_pooling_list_weights():
    embeddings = {
        "cat": np.array([1.0, 0.0]),
        "dog": np.array([0.0, 1.0])
    }
    vec = weighted_pooling(["cat", "dog"], embeddings, weights=[0.25, 0.75], embedding_dim=2)
    assert np.allclose(vec, [0.25, 0.75])


def test_weighted_pooling_zero_weights():
    embeddings = {"cat": np.array([1.0, 0.0])}
    vec = weighted_pooling(["cat"], embeddings, weights={"cat": 0.0}, embedding_dim=2)
    assert np.allclose(vec, [0.0, 0.0])


def test_document_embedder_train_and_embed(tmp_path):
    corpus = [
        ["python", "machine", "learning"],
        ["deep", "learning", "neural", "networks"],
        ["python", "code", "programming"]
    ]
    embedder = DocumentEmbedder(embedding_dim=16, seed=42)
    embedder.train_on_corpus(corpus, window_size=2)

    assert "python" in embedder.embeddings
    assert embedder.embeddings["python"].shape == (16,)

    doc_matrix = embedder.embed_corpus(corpus)
    assert doc_matrix.shape == (3, 16)

    # Test save and load
    save_file = tmp_path / "test_emb.npy"
    embedder.save_embeddings(save_file)
    assert save_file.exists()

    new_embedder = DocumentEmbedder(embedding_dim=16)
    new_embedder.load_embeddings(save_file)
    assert "python" in new_embedder.embeddings
    assert np.allclose(embedder.embeddings["python"], new_embedder.embeddings["python"])
