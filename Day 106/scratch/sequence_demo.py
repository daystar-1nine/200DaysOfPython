"""
Demonstration: Why word order matters in sequential text modeling.
Compares Bag-of-Words / Average Pooling with RNN hidden state trajectories.
"""

import numpy as np
from simple_rnn import forward_sequence


def mock_word_embeddings() -> dict:
    """Create deterministic mock word embeddings for clear illustration."""
    rng = np.random.RandomState(42)
    vocab = ["i", "like", "this", "movie", "not", "bad", "<PAD>", "<UNK>"]
    embeddings = {word: rng.randn(4).astype(np.float32) for word in vocab}
    return embeddings


def encode_sentence(words: list, emb_dict: dict) -> np.ndarray:
    """Convert word list to array of embedding vectors (seq_len, dim)."""
    return np.array([emb_dict.get(w, emb_dict["<UNK>"]) for w in words], dtype=np.float32)


def run_demonstration():
    print("=" * 65)
    print("DEMO: WORD ORDER SENSITIVITY (RNN vs POOLING)")
    print("=" * 65)

    emb_dict = mock_word_embeddings()
    dim = 4
    hidden_size = 6

    rng = np.random.RandomState(7)
    Wx = rng.randn(hidden_size, dim).astype(np.float32) * 0.2
    Wh = rng.randn(hidden_size, hidden_size).astype(np.float32) * 0.2
    b = np.zeros(hidden_size, dtype=np.float32)

    # Pair 1: Inverted sequence with identical vocabulary
    sent1 = ["not", "bad"]
    sent2 = ["bad", "not"]

    seq1 = encode_sentence(sent1, emb_dict)
    seq2 = encode_sentence(sent2, emb_dict)

    # 1. Bag of Words / Mean Pooling
    pool1 = seq1.mean(axis=0)
    pool2 = seq2.mean(axis=0)
    pooling_diff = np.linalg.norm(pool1 - pool2)

    # 2. RNN Forward Pass
    _, h_final_1 = forward_sequence(seq1, Wx, Wh, b)
    _, h_final_2 = forward_sequence(seq2, Wx, Wh, b)
    rnn_diff = np.linalg.norm(h_final_1 - h_final_2)

    print(f"\n1. Order Inversion Comparison: '{' '.join(sent1)}' vs '{' '.join(sent2)}'")
    print(f"   • Mean Pooling Vector Difference : {pooling_diff:.6f}  (Order-agnostic)")
    print(f"   • RNN Final Hidden State Difference : {rnn_diff:.6f}  (Order-sensitive)")
    assert np.isclose(pooling_diff, 0.0), "Mean pooling should be completely invariant to order!"
    assert rnn_diff > 0.01, "RNN final hidden states must differ for different token orderings!"

    # Pair 2: Negation injection
    sent3 = ["i", "like", "this", "movie"]
    sent4 = ["i", "not", "like", "this", "movie"]

    seq3 = encode_sentence(sent3, emb_dict)
    seq4 = encode_sentence(sent4, emb_dict)

    _, h_final_3 = forward_sequence(seq3, Wx, Wh, b)
    _, h_final_4 = forward_sequence(seq4, Wx, Wh, b)
    neg_diff = np.linalg.norm(h_final_3 - h_final_4)

    print(f"\n2. Negation Insertion Comparison: '{' '.join(sent3)}' vs '{' '.join(sent4)}'")
    print(f"   • RNN State Trajectory Shift    : {neg_diff:.6f}")
    print("\n[SUCCESS] Demonstrated why sequential recurrence captures syntax and word order!")
    print("=" * 65)


if __name__ == "__main__":
    run_demonstration()
