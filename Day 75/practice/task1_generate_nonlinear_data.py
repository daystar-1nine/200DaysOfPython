import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples=100):
    np.random.seed(42)
    X = np.random.uniform(-3, 3, n_samples)
    y = 0.5 * X**3 - 2 * X**2 + X + 10 + np.random.normal(0, 2, n_samples)
    return X.reshape(-1, 1), y

if __name__ == '__main__':
    X, y = generate_data()
    plt.scatter(X, y)
    plt.title("Non-linear Data")
    plt.savefig("task1_output.png")
