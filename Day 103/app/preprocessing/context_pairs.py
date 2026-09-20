"""Context window pair generation for Skip-Gram."""
from typing import List, Tuple
from app.data.vocabulary import Vocabulary

def generate_context_pairs(
    tokenized_doc: List[str],
    vocab: Vocabulary,
    window_size: int = 2
) -> List[Tuple[int, int]]:
    """
    Generates (target_id, context_id) pairs from a tokenized sentence
    using a symmetrical sliding context window.
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if len(tokenized_doc) < 2:
        return []

    # Map tokens to IDs
    ids = [vocab.get_id(token) for token in tokenized_doc if token in vocab]
    pairs = []

    for i, target_id in enumerate(ids):
        start = max(0, i - window_size)
        end = min(len(ids), i + window_size + 1)
        for j in range(start, end):
            if i != j:
                pairs.append((target_id, ids[j]))

    return pairs

def generate_all_pairs(
    tokenized_corpus: List[List[str]],
    vocab: Vocabulary,
    window_size: int = 2
) -> List[Tuple[int, int]]:
    """Generates all context pairs across the full corpus."""
    all_pairs = []
    for doc in tokenized_corpus:
        all_pairs.extend(generate_context_pairs(doc, vocab, window_size=window_size))
    return all_pairs
