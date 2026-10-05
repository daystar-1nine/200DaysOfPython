"""
Custom tokenizer with native support for atomic chat special tokens.
Converts text to integer token IDs and vice versa, treating roles and delimiters
as single atomic tokens.
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Union


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|unk|>"
]


class ChatTokenizer:
    """
    Character-level tokenizer extended with atomic multi-character chat tokens.
    Guarantees deterministic bidirectional encoding and decoding.
    """
    def __init__(self, chars: Optional[List[str]] = None):
        self.special_tokens = list(SPECIAL_TOKENS)
        self.pad_token = "<|pad|>"
        self.system_token = "<|system|>"
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"
        self.end_token = "<|end|>"
        self.unk_token = "<|unk|>"

        # Compile regex pattern to split text into special tokens vs regular text
        escaped_specials = [re.escape(tok) for tok in self.special_tokens]
        self._special_regex = re.compile(f"({'|'.join(escaped_specials)})")

        # Base vocabulary initialization
        self.itos: Dict[int, str] = {}
        self.stoi: Dict[str, int] = {}

        # 1. Assign special tokens starting at ID 0
        for idx, token in enumerate(self.special_tokens):
            self.itos[idx] = token
            self.stoi[token] = idx

        # 2. Add standard printable ASCII characters if not already provided
        default_chars = [chr(i) for i in range(32, 127)] + ["\n", "\t", "\r"]
        initial_chars = chars if chars is not None else default_chars

        current_id = len(self.special_tokens)
        for ch in sorted(list(set(initial_chars))):
            if ch not in self.stoi:
                self.itos[current_id] = ch
                self.stoi[ch] = current_id
                current_id += 1

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def system_id(self) -> int:
        return self.stoi[self.system_token]

    @property
    def user_id(self) -> int:
        return self.stoi[self.user_token]

    @property
    def assistant_id(self) -> int:
        return self.stoi[self.assistant_token]

    @property
    def end_id(self) -> int:
        return self.stoi[self.end_token]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    def encode(self, text: str) -> List[int]:
        """
        Encodes a string into a list of integer token IDs.
        Treats special tokens atomically, and maps other characters individually.
        """
        if not text:
            return []

        token_ids: List[int] = []
        segments = self._special_regex.split(text)

        for segment in segments:
            if not segment:
                continue
            if segment in self.stoi:
                # Matched special token
                token_ids.append(self.stoi[segment])
            else:
                # Regular characters
                for ch in segment:
                    token_ids.append(self.stoi.get(ch, self.unk_id))

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Decodes a list of integer token IDs back into a string.
        Optionally filters out special formatting tokens.
        """
        out_parts = []
        for tid in token_ids:
            token_str = self.itos.get(tid, self.unk_token)
            if skip_special_tokens and token_str in self.special_tokens:
                continue
            out_parts.append(token_str)
        return "".join(out_parts)

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves vocabulary mapping to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "special_tokens": self.special_tokens,
            "itos": {str(k): v for k, v in self.itos.items()},
            "stoi": self.stoi
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ChatTokenizer":
        """Loads vocabulary mapping from JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(chars=[])
        tokenizer.special_tokens = data["special_tokens"]
        tokenizer.itos = {int(k): v for k, v in data["itos"].items()}
        tokenizer.stoi = data["stoi"]
        escaped_specials = [re.escape(tok) for tok in tokenizer.special_tokens]
        tokenizer._special_regex = re.compile(f"({'|'.join(escaped_specials)})")
        return tokenizer
