import numpy as np

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def rbf_kernel(x1, x2, gamma=0.1):
    distance = np.linalg.norm(x1 - x2) ** 2
    return np.exp(-gamma * distance)

def decision_function(w, b, x):
    """Linear SVM Decision Function: f(x) = w^T x + b"""
    return np.dot(w, x) + b

def hinge_loss(w, b, X, y, C=1.0):
    """
    Hinge Loss for Linear SVM:
    L = 1/2 ||w||^2 + C * sum(max(0, 1 - y_i * (w^T x_i + b)))
    """
    margin = 1 - y * (np.dot(X, w) + b)
    margin = np.maximum(0, margin)
    regularization = 0.5 * np.dot(w, w)
    return regularization + C * np.sum(margin)

def test_svm_concepts():
    print("Testing SVM Mathematical Concepts...")
    
    # 1. Dot Product & Margin
    w = np.array([0.5, -0.5])
    b = 0.1
    x = np.array([2.0, 1.0])
    y_true = 1 # Class +1
    
    dec = decision_function(w, b, x)
    assert dec == 0.6
    
    # 2. Hinge Loss
    X = np.array([[2.0, 1.0], [0.5, 0.5]])
    y = np.array([1, -1])
    
    loss = hinge_loss(w, b, X, y, C=1.0)
    assert loss >= 0
    
    # 3. RBF Kernel
    x1 = np.array([1.0, 2.0])
    x2 = np.array([1.0, 2.0])
    assert rbf_kernel(x1, x2, gamma=1.0) == 1.0 # Identical points = 1.0
    
    x3 = np.array([10.0, 10.0])
    assert rbf_kernel(x1, x3, gamma=1.0) < 0.1 # Far points = approaches 0
    
    print("✅ SVM Concepts passed")

if __name__ == "__main__":
    test_svm_concepts()
