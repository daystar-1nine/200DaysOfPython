"""
Coding Challenge 3: Sequence Padding
Pads or truncates an integer sequence to max_length.
"""

from typing import List


def pad_sequence(sequence: List[int], max_length: int, pad_value: int = 0) -> List[int]:
    """Pad or truncate sequence to fixed length.
    
    Args:
        sequence: List of integers.
        max_length: Desired length.
        pad_value: Padding token value.
        
    Returns:
        List of length max_length.
    """
    if len(sequence) >= max_length:
        return sequence[:max_length]
    return sequence + [pad_value] * (max_length - len(sequence))


if __name__ == "__main__":
    s1 = [2, 3, 4]
    s2 = [1, 2, 3, 4, 5, 6, 7]
    p1 = pad_sequence(s1, 5)
    p2 = pad_sequence(s2, 5)

    print("Challenge 3: Sequence Padding")
    print(f"  Short sequence: {s1} -> Padded: {p1}")
    print(f"  Long sequence : {s2} -> Truncated: {p2}")
    assert p1 == [2, 3, 4, 0, 0]
    assert p2 == [1, 2, 3, 4, 5]
    print("  [SUCCESS] Padding & truncation verified!")
