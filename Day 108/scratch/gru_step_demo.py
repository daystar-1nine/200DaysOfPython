"""
Demonstration of GRU sequence processing using pure NumPy.
Verifies the exact tensor shapes specified in Section 14:
  input dimension = 32
  hidden size = 64
  sequence length = 20
Expected hidden-state output: (64,)
Expected sequence output: (20, 64)
"""
import numpy as np
from gru_equations import gru_step, gru_forward_sequence

def run_gru_step_demo():
    np.random.seed(42)
    input_dim = 32
    hidden_dim = 64
    seq_len = 20
    
    # Initialize random weights
    weights = {
        "Wz": np.random.randn(hidden_dim, input_dim) * 0.05,
        "Uz": np.random.randn(hidden_dim, hidden_dim) * 0.05,
        "bz": np.zeros(hidden_dim),
        "Wr": np.random.randn(hidden_dim, input_dim) * 0.05,
        "Ur": np.random.randn(hidden_dim, hidden_dim) * 0.05,
        "br": np.zeros(hidden_dim),
        "Wh": np.random.randn(hidden_dim, input_dim) * 0.05,
        "Uh": np.random.randn(hidden_dim, hidden_dim) * 0.05,
        "bh": np.zeros(hidden_dim),
    }
    
    # Single step test
    x_t = np.random.randn(input_dim)
    h_prev = np.zeros(hidden_dim)
    h_next, gates = gru_step(
        x_t, h_prev,
        weights["Wz"], weights["Uz"], weights["bz"],
        weights["Wr"], weights["Ur"], weights["br"],
        weights["Wh"], weights["Uh"], weights["bh"]
    )
    
    print("=== GRU Step Demo ===")
    print(f"Input x_t shape:         {x_t.shape}")
    print(f"h_prev shape:            {h_prev.shape}")
    print(f"Update gate z_t shape:   {gates['z_t'].shape}")
    print(f"Reset gate r_t shape:    {gates['r_t'].shape}")
    print(f"Candidate h_tilde shape: {gates['h_tilde'].shape}")
    print(f"New h_t shape:           {h_next.shape}")
    assert h_next.shape == (hidden_dim,), f"Expected ({hidden_dim},), got {h_next.shape}"
    
    # Sequence test
    X = np.random.randn(seq_len, input_dim)
    h_0 = np.zeros(hidden_dim)
    all_h, h_final = gru_forward_sequence(X, h_0, weights)
    
    print("\n=== GRU Sequence Demo ===")
    print(f"Sequence input X shape:  {X.shape}")
    print(f"Complete sequence shape: {all_h.shape}")
    print(f"Final hidden state shape:{h_final.shape}")
    assert all_h.shape == (seq_len, hidden_dim), f"Expected ({seq_len}, {hidden_dim}), got {all_h.shape}"
    assert h_final.shape == (hidden_dim,), f"Expected ({hidden_dim},), got {h_final.shape}"
    print("\nAll shape assertions passed successfully!")

if __name__ == "__main__":
    run_gru_step_demo()
