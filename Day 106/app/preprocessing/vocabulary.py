"""
Vocabulary module for Day 106: RNNs & Sequential Text Learning.
Constructs deterministic token-to-integer mappings strictly on training data.
"""

from collections import Counter
import json
from pathlib import Path
from typing import Dict, Iterable, List, Union
from .tokenizer import WordTokenizer


class Vocabulary:
    """Vocabulary indexer with special tokens <PAD>=0 and <UNK>=1."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_ID = 0
    UNK_ID = 1

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID,
        }
        self.idx2word: Dict[int, str] = {
            self.PAD_ID: self.PAD_TOKEN,
            self.UNK_ID: self.UNK_TOKEN,
        }
        self.word_counts: Counter = Counter()
        self.is_fitted: bool = False

    def __len__(self) -> int:
        return len(self.word2idx)

    def fit(self, texts: Iterable[str]) -> "Vocabulary":
        """Build vocabulary from an iterable of training texts strictly.
        
        Args:
            texts: Training corpus.
            
        Returns:
            Fitted Vocabulary instance.
        """
        self.word_counts = Counter()
        for text in texts:
            tokens = WordTokenizer.tokenize(text)
            self.word_counts.update(tokens)

        # Reset dictionaries to special tokens
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID,
        }
        self.idx2word = {
            self.PAD_ID: self.PAD_TOKEN,
            self.UNK_ID: self.UNK_TOKEN,
        }

        # Assign indices sorted by descending frequency, then lexicographically for determinism
        next_idx = 2
        sorted_words = sorted(
            [w for w, c in self.word_counts.items() if c >= self.min_freq],
            key=lambda w: (-self.word_counts[w], w)
        )

        for word in sorted_words:
            self.word2idx[word] = next_idx
            self.idx2word[next_idx] = word
            next_idx += 1

        self.is_fitted = True
        return self

    def token_to_id(self, token: str) -> int:
        """Convert a single token to integer ID. Maps unseen tokens to UNK."""
        token_clean = token.strip().lower()
        return self.word2idx.get(token_clean, self.UNK_ID)

    def id_to_token(self, idx: int) -> str:
        """Convert an integer ID back to token string."""
        return self.idx2word.get(idx, self.UNK_TOKEN)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize vocabulary mappings to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "min_freq": self.min_freq,
            "word2idx": self.word2idx,
            "idx2word": {str(k): v for k, v in self.idx2word.items()},
            "word_counts": dict(self.word_counts.most_common(500)),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Vocabulary":
        """Deserialize vocabulary from JSON."""
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(min_freq=data.get("min_freq", 2))
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        vocab.word_counts = Counter(data.get("word_counts", {}))
        vocab.is_fitted = True
        return vocab
