"""
ChatTokenizer with atomic delimiter tokens and sub-character vocabulary management.
"""
from pathlib import Path
from typing import List, Dict, Optional
import json

VOCAB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "vocab.json"


class ChatTokenizer:
    """
    Tokenizer supporting atomic chat delimiter tokens and character fallback.
    """
    def __init__(self, vocab_path: Optional[Path] = None):
        self.vocab_path = vocab_path or VOCAB_PATH
        self.special_tokens = [
            "<|pad|>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|end|>",
            "<|unk|>"
        ]
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._load_vocab()

    def _load_vocab(self) -> None:
        if self.vocab_path and self.vocab_path.exists():
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "stoi" in data and "itos" in data:
                self.token_to_id = data["stoi"]
                self.id_to_token = {int(k): v for k, v in data["itos"].items()}
                return
            elif isinstance(data, dict):
                self.token_to_id = {k: int(v) for k, v in data.items() if not isinstance(v, list)}
                self.id_to_token = {v: k for k, v in self.token_to_id.items()}
                return

        self.token_to_id = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        # Standard ASCII printable characters
        for i in range(32, 127):
            ch = chr(i)
            if ch not in self.token_to_id:
                self.token_to_id[ch] = len(self.token_to_id)

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id.get("<|pad|>", 0)

    @property
    def system_id(self) -> int:
        return self.token_to_id.get("<|system|>", 1)

    @property
    def user_id(self) -> int:
        return self.token_to_id.get("<|user|>", 2)

    @property
    def assistant_id(self) -> int:
        return self.token_to_id.get("<|assistant|>", 3)

    @property
    def end_id(self) -> int:
        return self.token_to_id.get("<|end|>", 4)

    @property
    def unk_id(self) -> int:
        return self.token_to_id.get("<|unk|>", 5)

    def encode(self, text: str) -> List[int]:
        """Encodes text with atomic preservation of special delimiter tokens."""
        tokens: List[int] = []
        i = 0
        n = len(text)
        while i < n:
            matched_special = False
            for st in self.special_tokens:
                if text.startswith(st, i):
                    tokens.append(self.token_to_id[st])
                    i += len(st)
                    matched_special = True
                    break
            if not matched_special:
                ch = text[i]
                tokens.append(self.token_to_id.get(ch, self.unk_id))
                i += 1
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decodes token IDs into a reconstructed string."""
        chars: List[str] = []
        for tid in token_ids:
            tok = self.id_to_token.get(tid, "<|unk|>")
            if skip_special_tokens and tok in self.special_tokens:
                continue
            chars.append(tok)
        return "".join(chars)
