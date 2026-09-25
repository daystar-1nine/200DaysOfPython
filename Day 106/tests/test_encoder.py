"""
Tests for TextEncoder in Day 106: RNNs & Sequential Text Learning.
"""

from app.preprocessing.vocabulary import Vocabulary
from app.preprocessing.encoder import TextEncoder


def test_encoder_encode_single_sentence():
    vocab = Vocabulary(min_freq=1).fit(["win free prize"])
    encoder = TextEncoder(vocab)
    encoded = encoder.encode("win free prize")
    assert len(encoded) == 3
    assert all(isinstance(i, int) for i in encoded)
    assert all(i >= 2 for i in encoded)


def test_encoder_encode_unknown_tokens():
    vocab = Vocabulary(min_freq=1).fit(["win free"])
    encoder = TextEncoder(vocab)
    encoded = encoder.encode("win unseen_token")
    assert encoded[0] == vocab.token_to_id("win")
    assert encoded[1] == Vocabulary.UNK_ID


def test_encoder_encode_empty_text():
    vocab = Vocabulary(min_freq=1).fit(["win free"])
    encoder = TextEncoder(vocab)
    assert encoder.encode("") == []
    assert encoder.encode("   ") == []


def test_encoder_encode_batch():
    vocab = Vocabulary(min_freq=1).fit(["hello friend", "claim reward"])
    encoder = TextEncoder(vocab)
    batch = encoder.encode_batch(["hello friend", "claim reward", "unknown"])
    assert len(batch) == 3
    assert len(batch[0]) == 2
    assert len(batch[1]) == 2
    assert batch[2] == [Vocabulary.UNK_ID]


def test_encoder_decode():
    vocab = Vocabulary(min_freq=1).fit(["meet me later"])
    encoder = TextEncoder(vocab)
    ids = encoder.encode("meet me later")
    decoded = encoder.decode(ids)
    assert decoded == "meet me later"
