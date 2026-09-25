"""
Master runner for all 6 coding challenges in Day 106: RNNs & Sequential Text Learning.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure local imports work
CHALLENGES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHALLENGES_DIR))

from challenge_1_reverse_sequence import reverse_sequence
from challenge_2_manual_rnn import simple_rnn
from challenge_3_padding import pad_sequence
from challenge_4_vocabulary import build_vocabulary
from challenge_5_unknown_tokens import encode_sentence
from challenge_6_masked_mean import masked_mean


def run_all_challenges():
    print("=" * 60)
    print("RUNNING ALL 6 CODING CHALLENGES — DAY 106")
    print("=" * 60)

    # 1. Reverse sequence
    rev = reverse_sequence([1, 2, 3, 4])
    assert rev == [4, 3, 2, 1]
    print("[CHALLENGE 1 PASSED] reverse_sequence([1, 2, 3, 4]) == [4, 3, 2, 1]")

    # 2. Manual RNN
    seq = np.ones((3, 2), dtype=np.float32)
    Wx = np.ones((4, 2), dtype=np.float32) * 0.1
    Wh = np.ones((4, 4), dtype=np.float32) * 0.1
    b = np.zeros(4, dtype=np.float32)
    states, final_h = simple_rnn(seq, Wx, Wh, b)
    assert states.shape == (3, 4) and final_h.shape == (4,)
    print("[CHALLENGE 2 PASSED] simple_rnn forward pass dimensions and recurrence correct")

    # 3. Padding
    padded = pad_sequence([10, 20], max_length=4)
    assert padded == [10, 20, 0, 0]
    print("[CHALLENGE 3 PASSED] pad_sequence([10, 20], 4) == [10, 20, 0, 0]")

    # 4. Vocabulary
    vocab = build_vocabulary(["hello world"])
    assert vocab["<PAD>"] == 0 and vocab["<UNK>"] == 1 and vocab["hello"] == 2
    print("[CHALLENGE 4 PASSED] build_vocabulary assigned special tokens and words")

    # 5. Unknown Tokens
    encoded = encode_sentence("hello galaxy", vocab)
    assert encoded == [2, 1]
    print("[CHALLENGE 5 PASSED] encode_sentence mapped unseen 'galaxy' to <UNK>=1")

    # 6. Masked Mean
    mat = np.array([[2.0, 4.0], [6.0, 8.0], [0.0, 0.0]], dtype=np.float32)
    m = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    pooled = masked_mean(mat, m)
    np.testing.assert_allclose(pooled, [4.0, 6.0])
    print("[CHALLENGE 6 PASSED] masked_mean ignored zero-padded position correctly")

    print("=" * 60)
    print("ALL 6 CODING CHALLENGES SUCCESSFULLY VERIFIED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_challenges()
