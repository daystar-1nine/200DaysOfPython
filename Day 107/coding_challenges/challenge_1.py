import numpy as np

def lstm_forget_gate(x_t, h_prev, W_f, b_f):
    """
    Computes the forget gate activation.
    x_t: (batch, input_dim)
    h_prev: (batch, hidden_dim)
    W_f: (input_dim + hidden_dim, hidden_dim)
    b_f: (1, hidden_dim)
    """
    concat = np.hstack((x_t, h_prev))
    f_t = 1 / (1 + np.exp(-(np.dot(concat, W_f) + b_f)))
    return f_t

if __name__ == "__main__":
    np.random.seed(42)
    x = np.random.randn(2, 3)
    h = np.random.randn(2, 4)
    W = np.random.randn(7, 4)
    b = np.random.randn(1, 4)
    print(lstm_forget_gate(x, h, W, b))
