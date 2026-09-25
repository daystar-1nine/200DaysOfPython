"""
Tests for Vocabulary construction, special tokens, and indexing in Day 106.
"""

from pathlib import Path
from app.preprocessing.vocabulary import Vocabulary


def test_vocabulary_special_tokens():
    vocab = Vocabulary()
    assert Vocabulary.PAD_TOKEN in vocab.word2idx
    assert Vocabulary.UNK_TOKEN in vocab.word2idx
    assert vocab.word2idx[Vocabulary.PAD_TOKEN] == 0
    assert vocab.word2idx[Vocabulary.UNK_TOKEN] == 1


def test_vocabulary_fit_deterministic():
    texts = [
        "win cash prize",
        "win cash free lottery",
        "hello friend see you",
    ]
    v1 = Vocabulary(min_freq=1).fit(texts)
    v2 = Vocabulary(min_freq=1).fit(texts)
    assert v1.word2idx == v2.word2idx
    assert len(v1) == len(v2)


def test_vocabulary_min_freq_filtering():
    texts = [
        "apple banana orange",
        "apple banana",
        "apple pear",
    ]
    # 'apple' appears 3, 'banana' 2, 'orange' 1, 'pear' 1
    vocab = Vocabulary(min_freq=2).fit(texts)
    assert "apple" in vocab.word2idx
    assert "banana" in vocab.word2idx
    assert "orange" not in vocab.word2idx
    assert "pear" not in vocab.word2idx


def test_vocabulary_unknown_word_mapping():
    vocab = Vocabulary(min_freq=1).fit(["win", "free"])
    assert vocab.token_to_id("win") != vocab.UNK_ID
    assert vocab.token_to_id("unseen_word_xyz") == vocab.UNK_ID


def test_vocabulary_decode_and_id_to_token():
    vocab = Vocabulary(min_freq=1).fit(["urgent", "call"])
    urg_id = vocab.token_to_id("urgent")
    assert vocab.id_to_token(urg_id) == "urgent"
    assert vocab.id_to_token(99999) == Vocabulary.UNK_TOKEN


def test_vocabulary_serialization(tmp_path: Path):
    filepath = tmp_path / "vocab.json"
    vocab = Vocabulary(min_freq=1).fit(["claim", "prize", "cash"])
    vocab.save(filepath)
    assert filepath.exists()

    loaded_vocab = Vocabulary.load(filepath)
    assert loaded_vocab.word2idx == vocab.word2idx
    assert loaded_vocab.min_freq == vocab.min_freq
