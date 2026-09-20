"""Negative sampling distribution and generator."""
import numpy as np
from typing import List
from app.data.vocabulary import Vocabulary

class NegativeSampler:
    """
    Generates negative samples using the smoothed unigram distribution:
    P(w) proportional to count(w)^0.75
    """
    def __init__(self, vocab: Vocabulary, power: float = 0.75, table_size: int = 100000, seed: int = 42):
        self.vocab = vocab
        self.rng = np.random.default_rng(seed)
        
        # Compute smoothed distribution
        vocab_size = len(vocab)
        counts = np.array([vocab.word_counts[vocab.get_word(i)] for i in range(vocab_size)], dtype=np.float64)
        smoothed = counts ** power
        probs = smoothed / np.sum(smoothed)
        
        # Build discrete lookup table for O(1) sampling
        self.table = np.repeat(np.arange(vocab_size), np.round(probs * table_size).astype(int))
        if len(self.table) == 0:
            self.table = np.arange(vocab_size)

    def sample(self, num_samples: int, target_id: int, context_id: int) -> List[int]:
        """Draws negative samples excluding the target and true context word."""
        negatives = []
        vocab_size = len(self.vocab)
        if vocab_size <= 2:
            return [(target_id + 1) % max(1, vocab_size)] * num_samples

        attempts = 0
        max_attempts = num_samples * 20
        while len(negatives) < num_samples and attempts < max_attempts:
            attempts += 1
            sampled_idx = int(self.rng.choice(self.table))
            if sampled_idx != target_id and sampled_idx != context_id:
                negatives.append(sampled_idx)

        while len(negatives) < num_samples:
            sampled_idx = int(self.rng.integers(0, vocab_size))
            if sampled_idx != target_id:
                negatives.append(sampled_idx)
            else:
                negatives.append((target_id + 1) % vocab_size)

        return negatives
