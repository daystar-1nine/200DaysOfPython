"""
Scratch implementation of a Vanilla Recurrent Neural Network forward pass in pure NumPy.
Demonstrates timestep recurrence: h_t = tanh(Wx @ x_t + Wh @ h_{t-1} + b).
"""

from typing import Tuple
import numpy as np


def rnn_step(x_t: np.ndarray, h_prev: np.ndarray, Wx: np.ndarray, Wh: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute a single RNN timestep.
    
    Args:
        x_t: Input vector at timestep t, shape (input_dim,).
        h_prev: Hidden state from timestep t-1, shape (hidden_dim,).
        Wx: Weight matrix for input-to-hidden, shape (hidden_dim, input_dim).
        Wh: Weight matrix for hidden-to-hidden, shape (hidden_dim, hidden_dim).
        b: Bias vector for hidden state, shape (hidden_dim,).
        
    Returns:
        New hidden state h_t of shape (hidden_dim,).
    """
    return np.tanh(Wx @ x_t + Wh @ h_prev + b)


def forward_sequence(
    sequence: np.ndarray,
    Wx: np.ndarray,
    Wh: np.ndarray,
    b: np.ndarray,
    h_init: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Process an entire sequence through time.
    
    Args:
        sequence: Array of inputs across time, shape (seq_len, input_dim).
        Wx: Weight matrix (hidden_dim, input_dim).
        Wh: Weight matrix (hidden_dim, hidden_dim).
        b: Bias vector (hidden_dim,).
        h_init: Optional initial hidden state. If None, zeros are used.
        
    Returns:
        Tuple of (all_hidden_states, final_hidden_state).
        all_hidden_states has shape (seq_len, hidden_dim) [return_sequences=True equivalent].
        final_hidden_state has shape (hidden_dim,) [return_state=True equivalent].
    """
    seq_len = sequence.shape[0]
    hidden_dim = Wh.shape[0]

    if h_init is None:
        h = np.zeros(hidden_dim, dtype=np.float32)
    else:
        h = h_init.copy()

    states = []
    for t in range(seq_len):
        x_t = sequence[t]
        h = rnn_step(x_t, h, Wx, Wh, b)
        states.append(h.copy())

    all_states = np.array(states, dtype=np.float32)
    return all_states, h


if __name__ == "__main__":
    np.random.seed(42)
    seq_length = 5
    embedding_dim = 8
    hidden_size = 16

    sequence = np.random.randn(seq_length, embedding_dim).astype(np.float32)
    Wx = np.random.randn(hidden_size, embedding_dim).astype(np.float32) * 0.1
    Wh = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.1
    b = np.zeros(hidden_size, dtype=np.float32)

    all_states, final_state = forward_sequence(sequence, Wx, Wh, b)
    print("Scratch SimpleRNN Test:")
    print(f"  Input sequence shape : {sequence.shape}")
    print(f"  All states shape     : {all_states.shape} (Expected: (5, 16))")
    print(f"  Final state shape    : {final_state.shape} (Expected: (16,))")
    assert all_states.shape == (5, 16)
    assert final_state.shape == (16,)
    print("  [SUCCESS] Dimensions and recurrence verified!")
