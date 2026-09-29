"""
Unit tests for CharacterTokenizer, CharacterTokenizerScratch, and SimpleBPETokenizer.
"""
import pytest
import string
from app.tokenizer.char_tokenizer import CharacterTokenizer
from app.tokenizer.bpe_tokenizer import SimpleBPETokenizer
from scratch.char_tokenizer import CharacterTokenizerScratch


class TestCharacterTokenizer:
    def test_vocab_size_includes_unk(self):
        tok = CharacterTokenizer(chars=["a", "b", "c"], unk_token="<unk>")
        assert tok.vocab_size == 4
        assert "<unk>" in tok.vocab

    def test_encode_decode_roundtrip(self, char_tokenizer, sample_text):
        encoded = char_tokenizer.encode(sample_text)
        decoded = char_tokenizer.decode(encoded)
        assert decoded == sample_text

    def test_unknown_character_fallback(self):
        tok = CharacterTokenizer(chars=["a", "b"], unk_token="<unk>")
        encoded = tok.encode("abc")
        assert encoded[0] == tok.char2idx["a"]
        assert encoded[1] == tok.char2idx["b"]
        assert encoded[2] == tok.unk_idx

    def test_empty_string_encode_decode(self, char_tokenizer):
        assert char_tokenizer.encode("") == []
        assert char_tokenizer.decode([]) == ""

    def test_save_and_load_vocabulary(self, tmp_path, char_tokenizer, sample_text):
        vocab_file = tmp_path / "vocab.json"
        char_tokenizer.save(vocab_file)
        assert vocab_file.exists()

        loaded_tok = CharacterTokenizer.load(vocab_file)
        assert loaded_tok.vocab_size == char_tokenizer.vocab_size
        assert loaded_tok.encode(sample_text) == char_tokenizer.encode(sample_text)

    def test_from_text_factory_method(self):
        text = "banana"
        tok = CharacterTokenizer.from_text(text)
        assert set(tok.vocab).issuperset({"b", "a", "n", "<unk>"})
        assert tok.decode(tok.encode(text)) == text

    def test_tokenizer_special_characters(self):
        special_str = "Line 1\nLine 2\tEnd: 100%!"
        tok = CharacterTokenizer.from_text(special_str)
        assert tok.decode(tok.encode(special_str)) == special_str

    def test_tokenizer_unk_index_consistency(self):
        tok = CharacterTokenizer(chars=["x", "y"], unk_token="<unk>")
        assert tok.char2idx[tok.unk_token] == tok.unk_idx
        assert tok.idx2char[tok.unk_idx] == tok.unk_token

    def test_character_tokenizer_printable_ascii(self):
        ascii_chars = string.ascii_letters + string.digits + string.punctuation + " "
        tok = CharacterTokenizer.from_text(ascii_chars)
        assert tok.decode(tok.encode(ascii_chars)) == ascii_chars


class TestCharacterTokenizerScratch:
    def test_scratch_encode_decode(self):
        text = "machine learning"
        tok = CharacterTokenizerScratch(list(text))
        encoded = tok.encode(text)
        decoded = tok.decode(encoded)
        assert decoded == text
        assert tok.vocab_size == len(set(text))

    def test_scratch_ignores_unseen_chars(self):
        tok = CharacterTokenizerScratch(["a", "b"])
        assert tok.encode("abz") == [tok.char2idx["a"], tok.char2idx["b"]]


class TestSimpleBPETokenizer:
    def test_bpe_fit_and_tokenize(self):
        text = "low lower newest widest lowest"
        bpe = SimpleBPETokenizer(num_merges=10)
        bpe.fit(text)
        tokens = bpe.tokenize("lowest")
        assert len(tokens) > 0
        assert "".join(tokens) == "lowest"

    def test_bpe_preserves_characters_on_zero_merges(self):
        text = "cat dog"
        bpe = SimpleBPETokenizer(num_merges=0)
        bpe.fit(text)
        assert bpe.tokenize_word("cat") == ["c", "a", "t"]

    def test_bpe_merges_reduce_token_count(self):
        text = "banana banana banana banana banana"
        bpe = SimpleBPETokenizer(num_merges=4)
        bpe.fit(text)
        tokens = bpe.tokenize_word("banana")
        # Raw characters: 6. After merges, should be fewer than 6 tokens!
        assert len(tokens) < 6
        assert "".join(tokens) == "banana"

    def test_bpe_unknown_word_segmentation(self):
        text = "low lower newest lowest"
        bpe = SimpleBPETokenizer(num_merges=5)
        bpe.fit(text)
        tokens = bpe.tokenize("widest")
        assert len(tokens) > 0
        assert "".join(tokens) == "widest"
