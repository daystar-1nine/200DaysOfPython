import pytest
from app.preprocessing.vocabulary import Vocabulary
from app.config import PAD_TOKEN, UNK_TOKEN, PAD_IDX, UNK_IDX

def test_vocabulary_initialization():
    vocab = Vocabulary(max_size=100)
    assert not vocab.is_fitted
    assert len(vocab) == 2
    assert vocab.get_idx(PAD_TOKEN) == PAD_IDX
    assert vocab.get_idx(UNK_TOKEN) == UNK_IDX
    assert vocab.get_word(PAD_IDX) == PAD_TOKEN
    assert vocab.get_word(UNK_IDX) == UNK_TOKEN

def test_vocabulary_fit():
    corpus = [
        ["urgent", "call", "now"],
        ["call", "me", "later"],
        ["urgent", "update"]
    ]
    vocab = Vocabulary(max_size=50)
    vocab.fit(corpus)
    assert vocab.is_fitted
    assert "urgent" in vocab.word2idx
    assert "call" in vocab.word2idx
    assert vocab.get_idx("urgent") >= 2
    assert vocab.get_idx("nonexistent_word_xyz") == UNK_IDX

def test_vocabulary_max_size_capping():
    corpus = [
        [f"word_{i}" for i in range(100)]
    ]
    vocab = Vocabulary(max_size=10)
    vocab.fit(corpus)
    assert len(vocab) <= 10

def test_vocabulary_zero_data_leakage():
    """
    CRITICAL LEAKAGE TEST:
    Verifies that a token exclusive to the test set NEVER enters the training vocabulary.
    """
    train_tokens = [["apple", "banana", "cherry"], ["apple", "cherry"]]
    test_tokens = [["secret_test_token_12345", "banana"]]
    
    vocab = Vocabulary(max_size=100)
    vocab.fit(train_tokens)
    
    assert "secret_test_token_12345" not in vocab.word2idx
    assert vocab.get_idx("secret_test_token_12345") == UNK_IDX
    assert "apple" in vocab.word2idx
    assert "cherry" in vocab.word2idx

@pytest.mark.parametrize("query_word, expected_known", [
    ("hello", True),
    ("world", True),
    ("mars", False),
    ("jupiter", False),
    ("<PAD>", True),
    ("<UNK>", True)
])
def test_vocabulary_queries(query_word, expected_known):
    vocab = Vocabulary(max_size=50)
    vocab.fit([["hello", "world"]])
    idx = vocab.get_idx(query_word)
    if expected_known:
        if query_word == "<PAD>":
            assert idx == PAD_IDX
        elif query_word == "<UNK>":
            assert idx == UNK_IDX
        else:
            assert idx >= 2
    else:
        assert idx == UNK_IDX
