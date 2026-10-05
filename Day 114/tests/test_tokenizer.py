"""
Unit tests for ChatTokenizer special token handling and encoding/decoding.
"""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from app.data.tokenizer import ChatTokenizer, SPECIAL_TOKENS


class TestChatTokenizer:
    def test_special_tokens_initialized(self, tokenizer):
        assert tokenizer.pad_id == tokenizer.stoi["<|pad|>"]
        assert tokenizer.system_id == tokenizer.stoi["<|system|>"]
        assert tokenizer.user_id == tokenizer.stoi["<|user|>"]
        assert tokenizer.assistant_id == tokenizer.stoi["<|assistant|>"]
        assert tokenizer.end_id == tokenizer.stoi["<|end|>"]
        assert tokenizer.unk_id == tokenizer.stoi["<|unk|>"]

    def test_special_tokens_are_unique(self, tokenizer):
        ids = [tokenizer.stoi[tok] for tok in SPECIAL_TOKENS]
        assert len(ids) == len(set(ids))

    def test_special_tokens_encoded_atomically(self, tokenizer):
        text = "<|user|>Explain Python<|end|>"
        encoded = tokenizer.encode(text)
        # Should start with 1 token for <|user|> and end with 1 token for <|end|>
        assert encoded[0] == tokenizer.user_id
        assert encoded[-1] == tokenizer.end_id

    def test_encode_empty_string(self, tokenizer):
        assert tokenizer.encode("") == []

    def test_decode_roundtrip_standard_text(self, tokenizer):
        original = "Hello World! 12345."
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)
        assert decoded == original

    def test_decode_roundtrip_with_chat_tokens(self, tokenizer):
        original = "<|system|>Sys<|end|><|user|>User<|end|><|assistant|>Ans<|end|>"
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)
        assert decoded == original

    def test_decode_skip_special_tokens(self, tokenizer):
        text = "<|user|>What is Python?<|end|>"
        encoded = tokenizer.encode(text)
        clean = tokenizer.decode(encoded, skip_special_tokens=True)
        assert clean == "What is Python?"
        assert "<|user|>" not in clean
        assert "<|end|>" not in clean

    def test_unknown_character_handling(self, tokenizer):
        # Character not in standard ASCII
        encoded = tokenizer.encode("©")
        assert tokenizer.unk_id in encoded

    def test_vocab_size_is_positive(self, tokenizer):
        assert tokenizer.vocab_size > len(SPECIAL_TOKENS)

    def test_save_and_load_tokenizer_roundtrip(self, tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            v_path = Path(tmpdir) / "vocab.json"
            tokenizer.save(v_path)
            assert v_path.exists()

            loaded_tok = ChatTokenizer.load(v_path)
            assert loaded_tok.vocab_size == tokenizer.vocab_size
            assert loaded_tok.special_tokens == tokenizer.special_tokens
            assert loaded_tok.encode("Test") == tokenizer.encode("Test")
            assert loaded_tok.encode("<|assistant|>") == [tokenizer.assistant_id]
