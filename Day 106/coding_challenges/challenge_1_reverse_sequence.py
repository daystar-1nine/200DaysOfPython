"""
Coding Challenge 1: Reverse Sequence
Given a list or array sequence, reverse its temporal order without external libraries.
"""

from typing import List, Any


def reverse_sequence(seq: List[Any]) -> List[Any]:
    """Reverse sequence order.
    
    Args:
        seq: Input list [x_1, x_2, ..., x_T].
        
    Returns:
        Reversed list [x_T, ..., x_2, x_1].
    """
    return seq[::-1]


if __name__ == "__main__":
    inp = [1, 2, 3, 4]
    out = reverse_sequence(inp)
    print("Challenge 1: Reverse Sequence")
    print(f"  Input : {inp}")
    print(f"  Output: {out}")
    assert out == [4, 3, 2, 1]
    print("  [SUCCESS] Reverse sequence verified!")
