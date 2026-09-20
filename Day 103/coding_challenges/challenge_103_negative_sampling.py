"""Challenge 6: Negative sampling generator."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

def sample_negatives(vocab_size: int, num_samples: int, target: int, context: int, rng=None) -> list:
    if rng is None:
        rng = np.random.default_rng(42)
    negatives = []
    while len(negatives) < num_samples:
        candidate = int(rng.integers(0, vocab_size))
        if candidate != target and candidate != context:
            negatives.append(candidate)
    return negatives

def run():
    print("=== CHALLENGE 103: NEGATIVE SAMPLING ===")
    negs = sample_negatives(vocab_size=10, num_samples=3, target=1, context=2)
    print("Sampled negatives for target=1, context=2:", negs)
    assert len(negs) == 3
    assert 1 not in negs and 2 not in negs
    print("[SUCCESS] Negative sampling verified!")

if __name__ == "__main__":
    run()
