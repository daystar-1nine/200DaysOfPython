"""
Vocabulary module for Day 105: Neural NLP & Text Classification.
Builds bidirectional token-to-integer mapping with special tokens.
"""

from collections import Counter
import json
from pathlib import Path
from typing import Dict, List, Optional, Union


class Vocabulary:
    """Manages token-to-integer mappings, special tokens, and frequency pruning."""

    def __init__(
        self,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        pad_id: int = 0,
        unk_id: int = 1
    ):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_id = pad_id
        self.unk_id = unk_id

        self.word2idx: Dict[str, int] = {self.pad_token: self.pad_id, self.unk_token: self.unk_id}
        self.idx2word: Dict[int, str] = {self.pad_id: self.pad_token, self.unk_id: self.unk_token}
        self.word_counts: Counter = Counter()
        self.is_fitted: bool = False

    def fit(self, tokenized_texts: List[List[str]], min_freq: int = 1) -> "Vocabulary":
        """Build vocabulary from tokenized training documents.
        
        Args:
            tokenized_texts: List of token lists.
            min_freq: Minimum frequency threshold to retain words.
            
        Returns:
            self
        """
        self.word_counts = Counter(word for doc in tokenized_texts for word in doc)

        # Sort alphabetically for determinism
        sorted_words = sorted([w for w, count in self.word_counts.items() if count >= min_freq])

        # Assign indices starting after special tokens
        next_id = max(self.pad_id, self.unk_id) + 1
        for word in sorted_words:
            if word not in self.word2idx:
                self.word2idx[word] = next_id
                self.idx2word[next_id] = word
                next_id += 1

        self.is_fitted = True
        return self

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert a list of word tokens into integer IDs.
        
        Args:
            tokens: List of tokens.
            
        Returns:
            List of integer IDs.
        """
        return [self.word2idx.get(token, self.unk_id) for token in tokens]

    def decode(self, ids: List[int], skip_special: bool = False) -> List[str]:
        """Convert a list of integer IDs back to word tokens.
        
        Args:
            ids: List of integer IDs.
            skip_special: Whether to omit <PAD> and <UNK> tokens.
            
        Returns:
            List of word strings.
        """
        words = []
        for idx in ids:
            w = self.idx2word.get(idx, self.unk_token)
            if skip_special and (w == self.pad_token or w == self.unk_token):
                continue
            words.append(w)
        return words

    def __len__(self) -> int:
        return len(self.word2idx)

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save vocabulary to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "word2idx": self.word2idx,
            "word_counts": dict(self.word_counts)
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Vocabulary":
        """Load vocabulary from JSON."""
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        vocab = cls(
            pad_token=payload["pad_token"],
            unk_token=payload["unk_token"],
            pad_id=payload["pad_id"],
            unk_id=payload["unk_id"]
        )
        vocab.word2idx = payload["word2idx"]
        vocab.idx2word = {int(idx): word for word, idx in vocab.word2idx.items()}
        vocab.word_counts = Counter(payload.get("word_counts", {}))
        vocab.is_fitted = True
        return vocab
