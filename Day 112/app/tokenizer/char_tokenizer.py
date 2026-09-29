"""
Character-level tokenizer for MiniGPT language modeling.
"""
from typing import List, Dict, Union
import json
from pathlib import Path


class CharacterTokenizer:
    """
    Maps individual characters to unique integer IDs and vice versa.
    Maintains a deterministic bidirectional mapping and handles vocabulary serialization.
    """
    def __init__(self, chars: List[str] = None, unk_token: str = "<unk>"):
        self.unk_token = unk_token
        if chars is not None:
            sorted_chars = sorted(list(set(chars)))
            if self.unk_token not in sorted_chars:
                self.vocab = [self.unk_token] + sorted_chars
            else:
                self.vocab = sorted_chars

            self.char2idx: Dict[str, int] = {ch: i for i, ch in enumerate(self.vocab)}
            self.idx2char: Dict[int, str] = {i: ch for i, ch in enumerate(self.vocab)}
            self.unk_idx = self.char2idx[self.unk_token]
        else:
            self.vocab = []
            self.char2idx = {}
            self.idx2char = {}
            self.unk_idx = 0

    @classmethod
    def from_text(cls, text: str, unk_token: str = "<unk>") -> "CharacterTokenizer":
        """Instantiates tokenizer by scanning all unique characters present in text."""
        unique_chars = sorted(list(set(text)))
        return cls(chars=unique_chars, unk_token=unk_token)

    @property
    def vocab_size(self) -> int:
        """Returns total number of tokens in the vocabulary."""
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        """Encodes a string into a list of integer token IDs."""
        return [self.char2idx.get(ch, self.unk_idx) for ch in text]

    def decode(self, ids: Union[List[int], List[float]]) -> str:
        """Decodes a list of token IDs back into a string sequence."""
        chars = []
        for idx in ids:
            int_id = int(idx)
            ch = self.idx2char.get(int_id, "")
            if ch != self.unk_token:
                chars.append(ch)
        return "".join(chars)

    def save(self, filepath: Path) -> None:
        """Saves vocabulary mapping to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab": self.vocab,
            "unk_token": self.unk_token
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "CharacterTokenizer":
        """Loads vocabulary mapping from JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(chars=data["vocab"], unk_token=data["unk_token"])
        return tok
