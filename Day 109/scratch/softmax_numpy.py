"""
Numerically stable Softmax implementation from scratch using NumPy.
"""
import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Computes numerically stable softmax along the specified axis.
    Subtracts the maximum value to prevent exp overflow.
    """
    # Numerical stability shift
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp_x

if __name__ == "__main__":
    scores = np.array([2.1, 0.5, 1.0])
    weights = softmax(scores)
    print("Scores:", scores)
    print("Softmax Weights:", weights)
    print("Sum of weights:", np.sum(weights))
    assert np.all(weights >= 0.0) and np.all(weights <= 1.0)
    assert np.isclose(np.sum(weights), 1.0)
    
    # Large numbers stability test
    large_scores = np.array([1000.0, 1001.0, 999.0])
    stable_weights = softmax(large_scores)
    print("Stable Weights for large scores:", stable_weights)
    assert not np.isnan(stable_weights).any()
    print("Softmax tests passed successfully!")
