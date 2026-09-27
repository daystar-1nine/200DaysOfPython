"""
Challenge 2: Implement gru_step without TensorFlow.
"""
import numpy as np
from challenge_1 import sigmoid

def gru_step(x, h_prev, weights):
    """
    weights dict contains:
      Wz, Uz, bz
      Wr, Ur, br
      Wh, Uh, bh
    """
    z = sigmoid(weights["Wz"] @ x + weights["Uz"] @ h_prev + weights["bz"])
    r = sigmoid(weights["Wr"] @ x + weights["Ur"] @ h_prev + weights["br"])
    candidate = np.tanh(weights["Wh"] @ x + weights["Uh"] @ (r * h_prev) + weights["bh"])
    h = (1.0 - z) * h_prev + z * candidate
    return h

if __name__ == "__main__":
    np.random.seed(42)
    in_dim, hid_dim = 16, 32
    weights = {
        "Wz": np.random.randn(hid_dim, in_dim)*0.1, "Uz": np.random.randn(hid_dim, hid_dim)*0.1, "bz": np.zeros(hid_dim),
        "Wr": np.random.randn(hid_dim, in_dim)*0.1, "Ur": np.random.randn(hid_dim, hid_dim)*0.1, "br": np.zeros(hid_dim),
        "Wh": np.random.randn(hid_dim, in_dim)*0.1, "Uh": np.random.randn(hid_dim, hid_dim)*0.1, "bh": np.zeros(hid_dim),
    }
    x_t = np.random.randn(in_dim)
    h_prev = np.zeros(hid_dim)
    h_next = gru_step(x_t, h_prev, weights)
    print("Computed h_next shape:", h_next.shape)
    assert h_next.shape == (hid_dim,)
