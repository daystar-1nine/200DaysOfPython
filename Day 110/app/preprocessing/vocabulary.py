"""
Vocabulary builder and token-to-index mapping ensuring zero data leakage.
"""
from collections import Counter
from typing import List, Dict, Iterable


class Vocabulary:
    def __init__(self, pad_token: str = "<PAD>", unk_token: str = "<UNK>", min_freq: int = 2):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.min_freq = min_freq

        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}

        # Reserved indices
        self.pad_idx = 0
        self.unk_idx = 1
        self._add_token(self.pad_token)
        self._add_token(self.unk_token)

    def _add_token(self, token: str) -> int:
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
            return idx
        return self.token2idx[token]

    def fit(self, tokenized_corpus: Iterable[List[str]]) -> "Vocabulary":
        """
        Builds vocabulary strictly from the training corpus.
        """
        counter = Counter()
        for tokens in tokenized_corpus:
            counter.update(tokens)

        for token, count in counter.most_common():
            if count >= self.min_freq:
                self._add_token(token)
        return self

    def encode(self, tokens: List[str]) -> List[int]:
        """Converts token list to sequence of integer IDs."""
        return [self.token2idx.get(t, self.unk_idx) for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """Converts sequence of integer IDs back to tokens."""
        return [self.idx2token.get(idx, self.unk_token) for idx in indices]

    def __len__(self) -> int:
        return len(self.token2idx)
