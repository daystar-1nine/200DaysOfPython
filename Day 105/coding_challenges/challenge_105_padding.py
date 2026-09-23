"""
Coding Challenges 105.4 & 105.5: Sequence Padding and Truncation.
Day 105: Neural NLP & Text Classification.
"""

from typing import List, Tuple


def challenge_pad_sequence(
    sequence: List[int],
    max_length: int,
    pad_id: int = 0
) -> Tuple[List[int], List[int]]:
    """Challenge 4 & 5: Implement sequence padding and truncation.
    
    If length < max_length, pad with pad_id.
    If length > max_length, truncate from the end.
    Returns (padded_sequence, mask).
    """
    seq = list(sequence)
    if len(seq) > max_length:
        truncated = seq[:max_length]
        return truncated, [1] * max_length
    else:
        pad_len = max_length - len(seq)
        padded = seq + [pad_id] * pad_len
        mask = [1] * len(seq) + [0] * pad_len
        return padded, mask


if __name__ == "__main__":
    # Test Challenge 4: Padding shorter sequence
    seq1 = [1, 2, 3]
    padded1, mask1 = challenge_pad_sequence(seq1, max_length=5, pad_id=0)
    print("Input:", seq1)
    print("Padded to 5:", padded1)
    print("Mask:", mask1)
    assert padded1 == [1, 2, 3, 0, 0]
    assert mask1 == [1, 1, 1, 0, 0]

    # Test Challenge 5: Truncation
    seq2 = [1, 2, 3, 4, 5, 6, 7]
    padded2, mask2 = challenge_pad_sequence(seq2, max_length=4, pad_id=0)
    print("Input:", seq2)
    print("Truncated to 4:", padded2)
    print("Mask:", mask2)
    assert padded2 == [1, 2, 3, 4]
    assert mask2 == [1, 1, 1, 1]

    print("[SUCCESS] Challenges 105.4 & 105.5 Passed!")
