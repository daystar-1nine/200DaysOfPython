"""Practice 3: Mathematical Proof of Ensemble Error Reduction."""
import math

def nCr(n, r):
    return math.comb(n, r)

def ensemble_error_probability(n_classifiers: int, base_error_rate: float) -> float:
    """Compute probability that a majority of independent classifiers make an error.
    Using Binomial CDF: sum_{k=ceil((n+1)/2)}^n C(n, k) * (p)^k * (1-p)^(n-k)
    """
    majority = (n_classifiers // 2) + 1
    total_error = 0.0
    for k in range(majority, n_classifiers + 1):
        prob_k = nCr(n_classifiers, k) * (base_error_rate ** k) * ((1.0 - base_error_rate) ** (n_classifiers - k))
        total_error += prob_k
    return total_error

if __name__ == '__main__':
    base_p = 0.35  # Individual tree error rate (65% accuracy)
    tree_counts = [1, 5, 11, 25, 51, 101]
    
    print(f"Base Classifier Error Rate: {base_p:.2f} (Accuracy: {1-base_p:.2f})")
    print("-" * 50)
    for m in tree_counts:
        err = ensemble_error_probability(m, base_p)
        acc = 1.0 - err
        print(f"Trees: {m:3d} | Ensemble Error: {err:8.5f} | Ensemble Accuracy: {acc*100:6.2f}%")
        
    err_101 = ensemble_error_probability(101, base_p)
    assert err_101 < 0.002
    print("Practice 3 passed!")
