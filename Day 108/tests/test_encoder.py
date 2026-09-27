import pytest
from app.preprocessing.vocabulary import Vocabulary
from app.preprocessing.encoder import encode_tokens, encode_texts
from app.preprocessing.tokenizer import tokenize_text
from app.config import UNK_IDX, PAD_IDX

@pytest.fixture
def fitted_vocab():
    v = Vocabulary()
    v.fit([["hello", "friend", "win", "prize"]])
    return v

def test_tokenize_text():
    tokens = tokenize_text("URGENT: Win $1000 prize now!")
    assert "urgent" in tokens
    assert "win" in tokens
    assert "currency" in tokens
    assert "prize" in tokens

def test_encode_tokens(fitted_vocab):
    tokens = ["hello", "unknown_word", "win"]
    encoded = encode_tokens(tokens, fitted_vocab)
    assert len(encoded) == 3
    assert encoded[0] == fitted_vocab.get_idx("hello")
    assert encoded[1] == UNK_IDX
    assert encoded[2] == fitted_vocab.get_idx("win")

def test_encode_texts(fitted_vocab):
    texts = ["hello friend", "unknown phrase here"]
    encoded = encode_texts(texts, fitted_vocab, tokenize_text)
    assert len(encoded) == 2
    assert len(encoded[0]) == 2
    assert encoded[1] == [UNK_IDX, UNK_IDX, UNK_IDX]

@pytest.mark.parametrize("text, expected_len", [
    ("one", 1),
    ("one two three", 3),
    ("", 0),
    ("hello, world!", 2),
    ("test-hyphen word", 3)
])
def test_tokenize_various_strings(text, expected_len):
    tokens = tokenize_text(text)
    assert len(tokens) == expected_len
