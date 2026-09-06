"""
Day 68 Task 7: Sampling Bias Experiment
Demonstrates how unrepresentative convenience sampling introduces
irrecoverable systematic error into parameter estimation.
"""
import numpy as np

def main():
    rng = np.random.default_rng(42)
    
    # 1. True Population (80% Regular users spending ~ $80, 20% Occasional spending ~ $25)
    reg_users = rng.normal(loc=80.0, scale=10.0, size=80_000)
    occ_users = rng.normal(loc=25.0, scale=8.0, size=20_000)
    population = np.concatenate([reg_users, occ_users])
    true_pop_mean = float(np.mean(population))
    
    # 2. Unbiased Simple Random Sample (n = 200)
    random_sample = rng.choice(population, size=200, replace=False)
    random_sample_mean = float(np.mean(random_sample))
    
    # 3. Biased Convenience Sample (e.g. surveyed only on premium app screen: 98% regular users)
    biased_sample = np.concatenate([
        rng.choice(reg_users, size=196, replace=False),
        rng.choice(occ_users, size=4, replace=False)
    ])
    biased_sample_mean = float(np.mean(biased_sample))
    
    print("=" * 70)
    print("  TASK 7: SAMPLING BIAS EXPERIMENT")
    print("=" * 70)
    print(f"True Population Mean (mu):       ${true_pop_mean:.2f}")
    print(f"Unbiased Random Sample Mean:     ${random_sample_mean:.2f} (Error: ${abs(random_sample_mean - true_pop_mean):.2f})")
    print(f"Biased Convenience Sample Mean:  ${biased_sample_mean:.2f} (Error: ${abs(biased_sample_mean - true_pop_mean):.2f})")
    print("-" * 70)
    print("Key Insight: The biased sample overestimates true population spend by over $10!")
    print("Increasing sample size of a biased sample will NEVER eliminate systematic bias.")

if __name__ == "__main__":
    main()
