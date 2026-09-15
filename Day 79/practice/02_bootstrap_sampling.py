"""Practice 2: Bootstrap Sampling with Replacement & OOB Proof."""
import numpy as np

def bootstrap_sample(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Draw bootstrap sample with replacement and identify Out-of-Bag (OOB) indices."""
    rng = np.random.RandomState(seed)
    n = len(X)
    boot_indices = rng.choice(n, size=n, replace=True)
    oob_mask = np.ones(n, dtype=bool)
    oob_mask[boot_indices] = False
    oob_indices = np.where(oob_mask)[0]
    return X[boot_indices], y[boot_indices], oob_indices

def theoretical_oob_ratio(n: int) -> float:
    """Theoretical probability an item is left out: (1 - 1/n)^n -> 1/e ~ 0.3679."""
    return float((1.0 - 1.0 / n) ** n)

if __name__ == '__main__':
    n_samples = 10000
    X = np.arange(n_samples)
    y = np.ones(n_samples)
    _, _, oob_idx = bootstrap_sample(X, y, seed=123)
    
    emp_ratio = len(oob_idx) / n_samples
    theo_ratio = theoretical_oob_ratio(n_samples)
    inv_e = 1.0 / np.e
    
    print(f"Empirical OOB Ratio:   {emp_ratio:.4f}")
    print(f"Theoretical (n={n_samples}): {theo_ratio:.4f}")
    print(f"Limit (1/e):           {inv_e:.4f}")
    assert abs(emp_ratio - inv_e) < 0.01
    print("Practice 2 passed!")
