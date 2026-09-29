"""
Unit tests for WordPiece subword tokenization and HuggingFace BertTokenizer integration.
"""
import pytest
from transformers import BertTokenizer
from scratch.wordpiece_demo import WordPieceTokenizerScratch
from app.preprocessing.tokenizer import inspect_tokenization


class TestWordPieceScratch:
    @pytest.fixture
    def scratch_tokenizer(self):
        vocab = ["play", "##ing", "##ed", "##er", "##ful", "un", "##believ", "##able", "the", "cat", "[UNK]"]
        return WordPieceTokenizerScratch(vocab=vocab, unk_token="[UNK]")

    def test_single_in_vocab_token(self, scratch_tokenizer):
        tokens = scratch_tokenizer.tokenize("play")
        assert tokens == ["play"]

    def test_subword_decomposition_with_hash(self, scratch_tokenizer):
        tokens = scratch_tokenizer.tokenize("playing")
        assert tokens == ["play", "##ing"]

    def test_multiple_subwords(self, scratch_tokenizer):
        tokens = scratch_tokenizer.tokenize("unbelievable")
        assert tokens == ["un", "##believ", "##able"]

    def test_unknown_token_fallback(self, scratch_tokenizer):
        tokens = scratch_tokenizer.tokenize("xylophone")
        assert tokens == ["[UNK]"]

    def test_word_reconstruction_from_subwords(self, scratch_tokenizer):
        subwords = ["un", "##believ", "##able"]
        reconstructed = "".join([s.replace("##", "") for s in subwords])
        assert reconstructed == "unbelievable"

    def test_empty_string_tokenization(self, scratch_tokenizer):
        tokens = scratch_tokenizer.tokenize("")
        assert tokens == []

    def test_vocab_size_property(self, scratch_tokenizer):
        assert scratch_tokenizer.vocab_size == 11


class TestBertTokenizerIntegration:
    def test_special_tokens_defined(self, tokenizer):
        assert tokenizer.cls_token == "[CLS]"
        assert tokenizer.sep_token == "[SEP]"
        assert tokenizer.pad_token == "[PAD]"
        assert tokenizer.mask_token == "[MASK]"
        assert tokenizer.unk_token == "[UNK]"

    def test_special_tokens_ids(self, tokenizer):
        assert tokenizer.cls_token_id == 101
        assert tokenizer.sep_token_id == 102
        assert tokenizer.pad_token_id == 0
        assert tokenizer.unk_token_id == 100
        assert tokenizer.mask_token_id == 103

    def test_tokenization_adds_cls_and_sep(self, tokenizer):
        encoded = tokenizer("Hello world", add_special_tokens=True)
        ids = encoded["input_ids"]
        assert ids[0] == tokenizer.cls_token_id
        assert ids[-1] == tokenizer.sep_token_id

    def test_tokenization_subword_hashes(self, tokenizer):
        tokens = tokenizer.tokenize("disproportionate")
        assert any(t.startswith("##") for t in tokens)

    def test_padding_to_max_length(self, tokenizer):
        max_len = 32
        encoded = tokenizer("Short sentence", padding="max_length", max_length=max_len)
        assert len(encoded["input_ids"]) == max_len
        assert len(encoded["attention_mask"]) == max_len
        assert encoded["input_ids"][-1] == tokenizer.pad_token_id
        assert encoded["attention_mask"][-1] == 0

    def test_truncation_to_max_length(self, tokenizer):
        long_text = "word " * 200
        max_len = 64
        encoded = tokenizer(long_text, truncation=True, max_length=max_len)
        assert len(encoded["input_ids"]) == max_len
        assert encoded["input_ids"][-1] == tokenizer.sep_token_id

    def test_attention_mask_alignment(self, tokenizer):
        encoded = tokenizer("Test sentence", padding="max_length", max_length=16)
        ids = encoded["input_ids"]
        mask = encoded["attention_mask"]
        for token_id, mask_val in zip(ids, mask):
            if token_id == tokenizer.pad_token_id:
                assert mask_val == 0
            else:
                assert mask_val == 1

    def test_sentence_pair_token_types(self, tokenizer):
        encoded = tokenizer("First sentence", "Second sentence")
        token_types = encoded["token_type_ids"]
        sep_indices = [i for i, id_ in enumerate(encoded["input_ids"]) if id_ == tokenizer.sep_token_id]
        assert len(sep_indices) == 2

        # First segment (A) up to first [SEP] should be 0
        assert all(t == 0 for t in token_types[:sep_indices[0] + 1])
        # Second segment (B) after first [SEP] should be 1
        assert all(t == 1 for t in token_types[sep_indices[0] + 1:])

    def test_inspect_tokenization_helper(self, tokenizer):
        info = inspect_tokenization("Congratulations", tokenizer)
        assert "tokens" in info
        assert "ids" in info
        assert "subword_count" in info
        assert info["tokens"][0] == "[CLS]"
        assert info["tokens"][-1] == "[SEP]"
