"""
Subword Byte-Pair Encoding (BPE) demonstration tokenizer.
"""
from typing import List, Dict, Tuple, Set
import re
from collections import defaultdict


class SimpleBPETokenizer:
    """
    Demonstrates Byte-Pair Encoding (BPE) iterative vocabulary learning and tokenization.
    """
    def __init__(self, num_merges: int = 50):
        self.num_merges = num_merges
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Set[str] = set()

    def _get_stats(self, vocab_counts: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        pairs = defaultdict(int)
        for word_tuple, freq in vocab_counts.items():
            for i in range(len(word_tuple) - 1):
                pairs[(word_tuple[i], word_tuple[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], vocab_counts: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        new_vocab = {}
        bigram = pair
        replacement = "".join(pair)
        for word_tuple, freq in vocab_counts.items():
            new_tuple = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == bigram[0] and word_tuple[i + 1] == bigram[1]:
                    new_tuple.append(replacement)
                    i += 2
                else:
                    new_tuple.append(word_tuple[i])
                    i += 1
            new_vocab[tuple(new_tuple)] = freq
        return new_vocab

    def fit(self, text: str) -> None:
        """Learns subword merge rules from text."""
        words = text.strip().split()
        vocab_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        for w in words:
            vocab_counts[tuple(list(w))] += 1

        for _ in range(self.num_merges):
            pairs = self._get_stats(vocab_counts)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            vocab_counts = self._merge_vocab(best_pair, vocab_counts)

        # Collect final vocabulary
        self.vocab = set()
        for word_tuple in vocab_counts.keys():
            self.vocab.update(word_tuple)

    def tokenize_word(self, word: str) -> List[str]:
        """Applies learned merge rules to tokenize a single word."""
        word_tuple = tuple(list(word))
        for pair in self.merges:
            replacement = "".join(pair)
            new_tuple = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == pair[0] and word_tuple[i + 1] == pair[1]:
                    new_tuple.append(replacement)
                    i += 2
                else:
                    new_tuple.append(word_tuple[i])
                    i += 1
            word_tuple = tuple(new_tuple)
        return list(word_tuple)

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        for w in text.strip().split():
            tokens.extend(self.tokenize_word(w))
        return tokens
