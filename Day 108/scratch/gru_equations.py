"""
Pure NumPy implementation of GRU equations from scratch.
"""
import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid activation function.
    Clips values to avoid floating overflow.
    """
    x_clipped = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def tanh(x: np.ndarray) -> np.ndarray:
    """
    Hyperbolic tangent activation function.
    """
    return np.tanh(x)

def gru_step(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    Wz: np.ndarray,
    Uz: np.ndarray,
    bz: np.ndarray,
    Wr: np.ndarray,
    Ur: np.ndarray,
    br: np.ndarray,
    Wh: np.ndarray,
    Uh: np.ndarray,
    bh: np.ndarray
):
    """
    Performs a single time-step forward pass through a GRU cell.
    
    Equations:
      z_t = sigma(Wz @ x_t + Uz @ h_prev + bz)               (Update Gate)
      r_t = sigma(Wr @ x_t + Ur @ h_prev + br)               (Reset Gate)
      h_tilde = tanh(Wh @ x_t + Uh @ (r_t * h_prev) + bh)    (Candidate Hidden State)
      h_t = (1 - z_t) * h_prev + z_t * h_tilde              (New Hidden State)
      
    Returns:
      h_t: New hidden state
      gates: Dict containing (z_t, r_t, h_tilde) for inspection and gradient tracing
    """
    # 1. Update gate
    z_t = sigmoid(Wz @ x_t + Uz @ h_prev + bz)
    
    # 2. Reset gate
    r_t = sigmoid(Wr @ x_t + Ur @ h_prev + br)
    
    # 3. Candidate hidden state
    h_tilde = tanh(Wh @ x_t + Uh @ (r_t * h_prev) + bh)
    
    # 4. Final hidden state interpolation
    h_t = (1.0 - z_t) * h_prev + z_t * h_tilde
    
    gates = {
        "z_t": z_t,
        "r_t": r_t,
        "h_tilde": h_tilde
    }
    return h_t, gates

def gru_forward_sequence(
    X: np.ndarray,
    h_0: np.ndarray,
    weights: dict
):
    """
    Forward pass through an entire sequence of inputs X of shape (T, input_dim).
    Returns:
      outputs: ndarray of shape (T, hidden_dim) containing h_t for all time-steps.
      h_final: final hidden state h_T of shape (hidden_dim,).
    """
    T, _ = X.shape
    hidden_dim = h_0.shape[0]
    outputs = np.zeros((T, hidden_dim))
    
    h_current = h_0.copy()
    for t in range(T):
        x_t = X[t]
        h_current, _ = gru_step(
            x_t=x_t,
            h_prev=h_current,
            Wz=weights["Wz"],
            Uz=weights["Uz"],
            bz=weights["bz"],
            Wr=weights["Wr"],
            Ur=weights["Ur"],
            br=weights["br"],
            Wh=weights["Wh"],
            Uh=weights["Uh"],
            bh=weights["bh"]
        )
        outputs[t] = h_current
        
    return outputs, h_current
