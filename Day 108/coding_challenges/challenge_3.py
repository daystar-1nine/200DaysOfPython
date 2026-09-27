"""
Challenge 3: Process sequence [x1, x2, x3, x4, x5] through NumPy GRU.
Return [h1, h2, h3, h4, h5].
"""
import numpy as np
from challenge_2 import gru_step

def process_sequence(seq_inputs, h_init, weights):
    """
    seq_inputs: list of 5 input vectors [x1, x2, x3, x4, x5]
    returns: list of hidden states [h1, h2, h3, h4, h5]
    """
    h_history = []
    h_curr = h_init.copy()
    for x in seq_inputs:
        h_curr = gru_step(x, h_curr, weights)
        h_history.append(h_curr)
    return h_history

if __name__ == "__main__":
    np.random.seed(42)
    in_dim, hid_dim = 8, 16
    weights = {
        "Wz": np.random.randn(hid_dim, in_dim)*0.1, "Uz": np.random.randn(hid_dim, hid_dim)*0.1, "bz": np.zeros(hid_dim),
        "Wr": np.random.randn(hid_dim, in_dim)*0.1, "Ur": np.random.randn(hid_dim, hid_dim)*0.1, "br": np.zeros(hid_dim),
        "Wh": np.random.randn(hid_dim, in_dim)*0.1, "Uh": np.random.randn(hid_dim, hid_dim)*0.1, "bh": np.zeros(hid_dim),
    }
    inputs = [np.random.randn(in_dim) for _ in range(5)]
    h0 = np.zeros(hid_dim)
    h_states = process_sequence(inputs, h0, weights)
    print(f"Processed sequence of length {len(inputs)}, returned {len(h_states)} hidden states.")
    assert len(h_states) == 5
    assert h_states[0].shape == (hid_dim,)
