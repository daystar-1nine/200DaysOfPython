"""
Challenge 7: Create a function top_attention_tokens(tokens, weights, k=5).
"""
import numpy as np
from typing import List, Tuple

def top_attention_tokens(tokens: List[str], weights: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
    valid_len = min(len(tokens), len(weights))
    paired = [(tokens[i], float(weights[i])) for i in range(valid_len)]
    paired.sort(key=lambda item: item[1], reverse=True)
    return paired[:k]

if __name__ == "__main__":
    tokens = ["win", "a", "free", "cash", "prize", "now"]
    weights = np.array([0.25, 0.02, 0.35, 0.15, 0.18, 0.05])
    top_3 = top_attention_tokens(tokens, weights, k=3)
    print("Top 3 tokens:", top_3)
    assert top_3[0] == ("free", 0.35)
    assert top_3[1] == ("win", 0.25)
    assert top_3[2] == ("prize", 0.18)
    print("Challenge 7 passed!")
