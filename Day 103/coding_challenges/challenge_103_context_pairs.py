"""Task & Challenge 4: Context pair generation."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from typing import List, Tuple

def generate_context_pairs(tokens: List[str], window_size: int = 1) -> List[Tuple[str, str]]:
    """Generates (target, context) word pairs for arbitrary window size."""
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    pairs = []
    for i, target in enumerate(tokens):
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        for j in range(start, end):
            if i != j:
                pairs.append((target, tokens[j]))
    return pairs

def run():
    print("=== CHALLENGE 103: CONTEXT PAIRS ===")
    text = "I love machine learning"
    tokens = text.split()
    pairs = generate_context_pairs(tokens, window_size=1)
    
    expected = [
        ("I", "love"),
        ("love", "I"),
        ("love", "machine"),
        ("machine", "love"),
        ("machine", "learning"),
        ("learning", "machine")
    ]
    print(f"Tokens: {tokens}")
    print(f"Generated Pairs (window=1): {pairs}")
    assert pairs == expected
    print("[SUCCESS] Context pair generation verified!")

if __name__ == "__main__":
    run()
