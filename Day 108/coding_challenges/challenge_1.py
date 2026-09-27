"""
Challenge 1: Implement sigmoid function.
"""
import numpy as np

def sigmoid(x):
    """
    Numerically stable sigmoid function.
    """
    x = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x))

if __name__ == "__main__":
    test_inputs = np.array([-10.0, 0.0, 10.0])
    print("Sigmoid outputs:", sigmoid(test_inputs))
    assert np.isclose(sigmoid(0.0), 0.5)
