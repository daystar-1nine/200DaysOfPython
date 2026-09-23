"""
Unit tests for Vocabulary class.
Day 105: Neural NLP & Text Classification.
"""

import pytest
from app.preprocessing.vocabulary import Vocabulary


def test_vocabulary_special_tokens():
    vocab = Vocabulary()
    assert vocab.pad_id == 0
    assert vocab.unk_id == 1
    assert vocab.word2idx[vocab.pad_token] == 0
    assert vocab.word2idx[vocab.unk_token] == 1
    assert vocab.idx2word[0] == vocab.pad_token
    assert vocab.idx2word[1] == vocab.unk_token


def test_vocabulary_fit():
    texts = [
        ["python", "is", "great"],
        ["machine", "learning", "python"]
    ]
    vocab = Vocabulary().fit(texts, min_freq=1)
    assert "python" in vocab.word2idx
    assert "machine" in vocab.word2idx
    assert vocab.word_counts["python"] == 2
    assert vocab.vocab_size == 7  # <PAD>, <UNK>, python, is, great, machine, learning


def test_vocabulary_min_freq_filtering():
    texts = [
        ["rare", "word", "common"],
        ["common", "token", "common"]
    ]
    vocab = Vocabulary().fit(texts, min_freq=2)
    assert "common" in vocab.word2idx
    assert "rare" not in vocab.word2idx


def test_vocabulary_encode_known_words():
    texts = [["hello", "world"]]
    vocab = Vocabulary().fit(texts)
    encoded = vocab.encode(["hello", "world"])
    assert encoded == [vocab.word2idx["hello"], vocab.word2idx["world"]]


def test_vocabulary_encode_unknown_words():
    texts = [["apple", "banana"]]
    vocab = Vocabulary().fit(texts)
    encoded = vocab.encode(["apple", "dragon"])
    assert encoded == [vocab.word2idx["apple"], vocab.unk_id]


def test_vocabulary_decode_basic():
    texts = [["deep", "learning"]]
    vocab = Vocabulary().fit(texts)
    ids = [vocab.word2idx["deep"], vocab.word2idx["learning"]]
    assert vocab.decode(ids) == ["deep", "learning"]


def test_vocabulary_decode_skip_special():
    texts = [["data"]]
    vocab = Vocabulary().fit(texts)
    ids = [0, vocab.word2idx["data"], 1]
    assert vocab.decode(ids, skip_special=True) == ["data"]


def test_vocabulary_save_load(tmp_path):
    texts = [["test", "serialization"]]
    vocab = Vocabulary().fit(texts)
    save_file = tmp_path / "vocab.json"
    vocab.save(save_file)
    assert save_file.exists()

    loaded = Vocabulary.load(save_file)
    assert loaded.vocab_size == vocab.vocab_size
    assert loaded.word2idx == vocab.word2idx
