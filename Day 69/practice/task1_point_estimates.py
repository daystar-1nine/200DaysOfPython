"""
Day 69 Task 1: Point Estimates & Standard Error
Calculates sample mean, sample standard deviation (ddof=1), and standard error.
"""
import numpy as np

def compute_point_estimates(data: np.ndarray) -> dict:
    if len(data) == 0:
        raise ValueError("Data array cannot be empty.")
        
    n = len(data)
    mean_val = float(np.mean(data))
    # Sample standard deviation uses ddof=1 (Bessel's correction)
    std_val = float(np.std(data, ddof=1)) if n > 1 else 0.0
    se_val = std_val / np.sqrt(n) if n > 0 else 0.0
    
    return {
        "sample_size": n,
        "sample_mean": round(mean_val, 4),
        "sample_std": round(std_val, 4),
        "standard_error": round(se_val, 4)
    }

def main():
    print("=" * 60)
    print("  TASK 1: POINT ESTIMATES & STANDARD ERROR")
    print("=" * 60)
    # E-commerce transaction sample (customer spending in INR)
    sample_data = np.array([2150.0, 2480.0, 2390.0, 2750.0, 2100.0, 2890.0, 2450.0, 2600.0, 2300.0, 2520.0])
    stats = compute_point_estimates(sample_data)
    
    print(f"Sample Size (n):          {stats['sample_size']}")
    print(f"Sample Mean (Point Est):  Rs. {stats['sample_mean']:.2f}")
    print(f"Sample Std Dev (s):       Rs. {stats['sample_std']:.2f}")
    print(f"Standard Error (SE):      Rs. {stats['standard_error']:.2f}")
    print("-" * 60)
    print("Insight: Standard Error quantifies the precision of our point estimate.")

if __name__ == "__main__":
    main()
