"""
Day 68 Task 8: Bootstrap Resampling on Small Sample Data
Performs 10,000 bootstrap resamples on an n = 10 dataset,
estimating the standard error and 95% empirical percentile confidence interval.
"""
import numpy as np

def main():
    rng = np.random.default_rng(42)
    data = np.array([12.0, 15.0, 18.0, 20.0, 21.0, 22.0, 25.0, 28.0, 30.0, 35.0])
    n = len(data)
    n_bootstrap = 10_000
    
    # Generate 10,000 bootstrap samples WITH replacement
    boot_means = np.array([
        np.mean(rng.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    
    orig_mean = float(np.mean(data))
    boot_mean = float(np.mean(boot_means))
    boot_se = float(np.std(boot_means, ddof=1))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    
    print("=" * 70)
    print(f"  TASK 8: NON-PARAMETRIC BOOTSTRAP RESAMPLING (B = {n_bootstrap:,})")
    print("=" * 70)
    print(f"Original Sample Mean:             {orig_mean:.2f}")
    print(f"Bootstrap Resample Mean:          {boot_mean:.2f}")
    print(f"Bootstrap Standard Error (SE):    {boot_se:.4f}")
    print(f"95% Percentile Confidence Interval:[{ci_lower:.2f}, {ci_upper:.2f}]")
    print("-" * 70)
    print("Bootstrap principle: The empirical sample distribution serves as a proxy")
    print("for the unseen population distribution without assuming normality.")

if __name__ == "__main__":
    main()
