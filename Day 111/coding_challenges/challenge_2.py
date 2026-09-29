"""
Challenge 2: Implement BERT's 80/10/10 Masked Language Modeling token replacement strategy.
- 15% of candidate tokens selected.
- 80% replaced with [MASK].
- 10% replaced with random token from vocabulary.
- 10% kept unchanged.
- Selected positions get original token ID in labels; unselected get -100.
"""
from typing import Tuple, List, Optional
import numpy as np


def apply_mlm_masking(
    token_ids: np.ndarray,
    mask_token_id: int = 103,
    vocab_size: int = 30522,
    mask_prob: float = 0.15,
    special_token_ids: Optional[List[int]] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    if special_token_ids is None:
        special_token_ids = [0, 101, 102, 103]

    rng = np.random.default_rng(seed)
    tokens = np.array(token_ids, copy=True)
    labels = np.full_like(tokens, fill_value=-100)

    candidate_mask = ~np.isin(tokens, special_token_ids)
    rand_scores = rng.random(tokens.shape)
    selected = candidate_mask & (rand_scores < mask_prob)

    if not np.any(selected) and np.any(candidate_mask):
        cands = np.argwhere(candidate_mask)
        selected[tuple(cands[rng.integers(0, len(cands))])] = True

    labels[selected] = tokens[selected]

    decision = rng.random(tokens.shape)
    # 80%: [MASK]
    mask_80 = selected & (decision < 0.8)
    tokens[mask_80] = mask_token_id

    # 10%: random token
    rand_10 = selected & (decision >= 0.8) & (decision < 0.9)
    tokens[rand_10] = rng.integers(104, vocab_size, size=tokens.shape)[rand_10]

    # 10%: unchanged
    return tokens, labels


if __name__ == "__main__":
    seq = np.array([101, 2023, 2003, 1037, 2742, 102])
    masked, lbls = apply_mlm_masking(seq, mask_prob=0.5, seed=42)
    print("Original:", seq)
    print("Masked:  ", masked)
    print("Labels:  ", lbls)
    assert masked.shape == seq.shape
    assert lbls[0] == -100 and lbls[-1] == -100  # [CLS] and [SEP] preserved
    print("Challenge 2 passed successfully!")
