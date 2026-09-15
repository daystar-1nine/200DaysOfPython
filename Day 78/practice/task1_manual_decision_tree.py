"""Task 1: Manual Decision Tree Impurity Calculation."""
import numpy as np

def calculate_gini(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)

def run_task():
    # Parent node: 50 Churn (1), 50 Stay (0)
    parent = np.array([1]*50 + [0]*50)
    gini_parent = calculate_gini(parent)
    
    # Candidate Split: Group A (45 Churn, 5 Stay), Group B (5 Churn, 45 Stay)
    group_a = np.array([1]*45 + [0]*5)
    group_b = np.array([1]*5 + [0]*45)
    
    gini_a = calculate_gini(group_a)
    gini_b = calculate_gini(group_b)
    
    n_total = len(parent)
    weighted_gini = (len(group_a) / n_total) * gini_a + (len(group_b) / n_total) * gini_b
    gini_gain = gini_parent - weighted_gini
    
    print("=== Manual Gini Impurity Computation ===")
    print(f"Parent Gini:        {gini_parent:.4f}")
    print(f"Group A Gini:       {gini_a:.4f} (45 Churn, 5 Stay)")
    print(f"Group B Gini:       {gini_b:.4f} (5 Churn, 45 Stay)")
    print(f"Weighted Child Gini:{weighted_gini:.4f}")
    print(f"Gini Reduction:     {gini_gain:.4f} (Impurity dramatically reduced!)")

if __name__ == "__main__":
    run_task()
