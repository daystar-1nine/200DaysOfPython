"""Task 4 & Challenge 5: N-gram generator."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from typing import List, Tuple

def generate_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Generates n-grams from a list of tokens."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if not tokens or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

def run_challenge():
    tokens = ["I", "love", "machine", "learning"]
    bigrams = generate_ngrams(tokens, 2)
    expected = [("I", "love"), ("love", "machine"), ("machine", "learning")]
    print(f"Tokens: {tokens}")
    print(f"Bigrams: {bigrams}")
    assert bigrams == expected
    print("[SUCCESS] N-grams challenge passed.")

if __name__ == "__main__":
    run_challenge()
