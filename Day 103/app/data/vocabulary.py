"""Vocabulary construction and bidirectional word/ID mapping."""
from collections import Counter
from typing import List, Dict, Optional

class Vocabulary:
    """Manages vocabulary mappings and term frequencies."""
    def __init__(self, min_count: int = 1):
        self.min_count = min_count
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()
        self.is_built: bool = False

    def build_vocab(self, tokenized_corpus: List[List[str]]) -> "Vocabulary":
        """Builds vocabulary from tokenized documents."""
        self.word_counts = Counter()
        for doc in tokenized_corpus:
            self.word_counts.update(doc)

        sorted_words = sorted([w for w, c in self.word_counts.items() if c >= self.min_count])
        self.word2idx = {word: idx for idx, word in enumerate(sorted_words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.is_built = True
        return self

    def __len__(self) -> int:
        return len(self.word2idx)

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx

    def get_id(self, word: str, default: Optional[int] = None) -> Optional[int]:
        return self.word2idx.get(word, default)

    def get_word(self, idx: int, default: Optional[str] = None) -> Optional[str]:
        return self.idx2word.get(idx, default)

    def get_frequency_distribution(self) -> Dict[str, int]:
        return {w: self.word_counts[w] for w in self.word2idx}
