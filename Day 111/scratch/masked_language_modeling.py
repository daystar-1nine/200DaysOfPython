"""
Masked Language Modeling (MLM) token masking implementation from scratch.
Follows the original BERT 80/10/10 masking protocol (Devlin et al., 2018).
"""
from typing import Tuple, List, Optional
import numpy as np


def mask_tokens(
    token_ids: np.ndarray,
    mask_token_id: int = 103,
    vocab_size: int = 30522,
    mask_probability: float = 0.15,
    special_token_ids: Optional[List[int]] = None,
    seed: Optional[int] = None,
    mask_prob: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies the BERT Masked Language Modeling strategy to input token IDs.

    Rules for selected 15% of tokens:
      - 80% of the time: replaced with [MASK] token
      - 10% of the time: replaced with random word token
      - 10% of the time: kept unchanged

    Labels:
      - Selected token positions contain the original token ID as target
      - Non-selected positions are set to -100 (ignored in CrossEntropy loss)

    Args:
        token_ids: 1D or 2D integer numpy array of token IDs
        mask_token_id: ID of the [MASK] token (default: 103)
        vocab_size: total vocabulary size for random replacements
        mask_probability: probability of selecting a token (default: 0.15)
        special_token_ids: list of tokens not eligible for masking ([CLS], [SEP], [PAD])
        seed: optional random seed for reproducibility
    Returns:
        masked_tokens: array matching token_ids shape with masking applied
        labels: array matching token_ids shape with targets for selected positions
    """
    if mask_prob is not None:
        mask_probability = mask_prob

    rng = np.random.default_rng(seed)
    tokens = np.array(token_ids, copy=True)
    labels = np.full_like(tokens, fill_value=-100)

    if special_token_ids is None:
        # Default BERT uncased: 0=[PAD], 101=[CLS], 102=[SEP]
        special_token_ids = [0, 101, 102, 103]

    # Create candidate mask (tokens that are NOT special tokens)
    candidate_mask = np.isin(tokens, special_token_ids, invert=True)

    # Random selection with mask_probability
    random_probs = rng.random(tokens.shape)
    selected_mask = candidate_mask & (random_probs < mask_probability)

    # Ensure at least one token is masked if candidates exist and none were selected
    if not np.any(selected_mask) and np.any(candidate_mask):
        candidate_indices = np.argwhere(candidate_mask)
        chosen = candidate_indices[rng.integers(0, len(candidate_indices))]
        selected_mask[tuple(chosen)] = True

    # Assign labels to selected positions
    labels[selected_mask] = tokens[selected_mask]

    # Determine 80 / 10 / 10 strategy for each selected token
    decision_probs = rng.random(tokens.shape)

    # 80%: replace with [MASK]
    mask_80 = selected_mask & (decision_probs < 0.8)
    tokens[mask_80] = mask_token_id

    # 10%: replace with random token (0.8 <= p < 0.9)
    random_10 = selected_mask & (decision_probs >= 0.8) & (decision_probs < 0.9)
    random_tokens = rng.integers(low=104, high=vocab_size, size=tokens.shape)
    tokens[random_10] = random_tokens[random_10]

    # 10%: keep unchanged (decision_probs >= 0.9) -> tokens remain as is

    return tokens, labels


mask_tokens_mlm = mask_tokens


if __name__ == "__main__":
    sample_ids = np.array([101, 2009, 2293, 6251, 102])  # [CLS] it was amazing [SEP]
    masked, lbls = mask_tokens(sample_ids, mask_token_id=103, mask_probability=0.5, seed=42)
    print("Original:", sample_ids)
    print("Masked:  ", masked)
    print("Labels:  ", lbls)
    assert masked.shape == sample_ids.shape
    assert lbls.shape == sample_ids.shape
    assert lbls[0] == -100 and lbls[-1] == -100  # [CLS] and [SEP] never masked
    print("MLM verification successful!")
