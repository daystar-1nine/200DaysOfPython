"""
Coding Challenge 2: Manual RNN Forward Pass without TensorFlow.
Implements h_t = tanh(Wx @ x_t + Wh @ h_{t-1} + b).
"""

from typing import Tuple
import numpy as np


def simple_rnn(sequence: np.ndarray, Wx: np.ndarray, Wh: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Execute manual SimpleRNN forward loop over sequence.
    
    Args:
        sequence: Array of shape (T, D).
        Wx: Weight matrix (H, D).
        Wh: Weight matrix (H, H).
        b: Bias vector (H,).
        
    Returns:
        Tuple of (all_hidden_states of shape (T, H), final_hidden_state of shape (H,)).
    """
    T = sequence.shape[0]
    H = Wh.shape[0]
    h = np.zeros(H, dtype=np.float32)
    states = []

    for t in range(T):
        x_t = sequence[t]
        h = np.tanh(Wx @ x_t + Wh @ h + b)
        states.append(h.copy())

    return np.array(states), h


if __name__ == "__main__":
    np.random.seed(42)
    T, D, H = 4, 3, 5
    seq = np.random.randn(T, D).astype(np.float32)
    Wx = np.random.randn(H, D).astype(np.float32) * 0.1
    Wh = np.random.randn(H, H).astype(np.float32) * 0.1
    b = np.zeros(H, dtype=np.float32)

    states, final_h = simple_rnn(seq, Wx, Wh, b)
    print("Challenge 2: Manual RNN Forward Pass")
    print(f"  All states shape : {states.shape} (Expected: (4, 5))")
    print(f"  Final state shape: {final_h.shape} (Expected: (5,))")
    assert states.shape == (4, 5)
    assert final_h.shape == (5,)
    assert np.allclose(states[-1], final_h)
    print("  [SUCCESS] Manual RNN verified!")
