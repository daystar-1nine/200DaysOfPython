import numpy as np

def linear_layer(X, W, b):
    """Z = XW + b"""
    return np.dot(X, W) + b

def relu(Z):
    """Activation: ReLU"""
    return np.maximum(0, Z)

def softmax(Z):
    """Softmax activation with numerical stability."""
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def categorical_cross_entropy(y_pred, y_true):
    """
    Categorical Cross-Entropy Loss: L = -sum(y_true * log(y_pred))
    Includes epsilon protection to prevent log(0).
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]
    return loss

def test_deep_learning_concepts():
    print("Testing Deep Learning Forward Propagation & Loss...")
    
    # 1. Forward Prop (Input -> Linear -> ReLU)
    X = np.array([[1.0, -2.0]])
    W = np.array([[0.5, 0.1], [-0.5, 0.8]])
    b = np.array([0.1, -0.1])
    
    Z = linear_layer(X, W, b)
    A = relu(Z)
    assert np.all(A >= 0)
    
    # 2. Softmax Output
    Z_out = np.array([[2.0, 1.0, 0.1]])
    A_out = softmax(Z_out)
    assert np.isclose(np.sum(A_out, axis=1), 1.0)
    
    # 3. Categorical Cross Entropy
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    y_pred_perfect = np.array([[0.99, 0.005, 0.005], [0.01, 0.98, 0.01]])
    y_pred_bad = np.array([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    
    loss_perfect = categorical_cross_entropy(y_pred_perfect, y_true)
    loss_bad = categorical_cross_entropy(y_pred_bad, y_true)
    
    assert loss_perfect < loss_bad
    
    print("✅ Forward Propagation & CCE Loss passed")

if __name__ == "__main__":
    test_deep_learning_concepts()
