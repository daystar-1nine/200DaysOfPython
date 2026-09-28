"""
Challenge 1: Implement stable softmax() without TensorFlow or PyTorch.
"""
import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax by subtracting the maximum value along the axis.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

if __name__ == "__main__":
    scores = np.array([2.5, 1.2, 0.3])
    probs = softmax(scores)
    print("Probabilities:", probs)
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    
    # Large numbers numerical stability
    large_scores = np.array([1000.0, 1001.0, 999.0])
    stable_probs = softmax(large_scores)
    print("Stable Probs on 1000+:", stable_probs)
    assert not np.isnan(stable_probs).any()
    print("Challenge 1 passed!")
