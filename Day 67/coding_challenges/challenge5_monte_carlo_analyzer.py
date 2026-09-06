"""
Day 67 Challenge 5: Universal Monte Carlo Distribution Analyzer
Implements compare_theory_vs_simulation() to benchmark theoretical parameters
against empirical Monte Carlo sampling across discrete and continuous laws.
"""
from typing import Literal
import numpy as np
import pandas as pd
from scipy import stats

def compare_theory_vs_simulation(
    distribution_name: Literal["bernoulli", "binomial", "uniform", "normal", "poisson"],
    parameters: dict,
    n_samples: int = 50_000,
    seed: int = 42
) -> dict:
    """
    Universal Monte Carlo benchmark comparing theoretical distribution properties
    with empirical statistics from n_samples random variates.
    """
    if distribution_name == "bernoulli":
        p = parameters["p"]
        rv = stats.bernoulli(p)
        samples = rv.rvs(size=n_samples, random_state=seed)
        theo_mean = float(rv.mean())
        theo_var = float(rv.var())
        
    elif distribution_name == "binomial":
        n, p = parameters["n"], parameters["p"]
        rv = stats.binom(n, p)
        samples = rv.rvs(size=n_samples, random_state=seed)
        theo_mean = float(rv.mean())
        theo_var = float(rv.var())
        
    elif distribution_name == "uniform":
        a, b = parameters["a"], parameters["b"]
        rv = stats.uniform(loc=a, scale=b - a)
        samples = rv.rvs(size=n_samples, random_state=seed)
        theo_mean = float(rv.mean())
        theo_var = float(rv.var())
        
    elif distribution_name == "normal":
        mu, sigma = parameters["mu"], parameters["sigma"]
        rv = stats.norm(loc=mu, scale=sigma)
        samples = rv.rvs(size=n_samples, random_state=seed)
        theo_mean = float(rv.mean())
        theo_var = float(rv.var())
        
    elif distribution_name == "poisson":
        lam = parameters["lambda"]
        rv = stats.poisson(mu=lam)
        samples = rv.rvs(size=n_samples, random_state=seed)
        theo_mean = float(rv.mean())
        theo_var = float(rv.var())
        
    else:
        raise ValueError(f"Unsupported distribution: {distribution_name}")
        
    emp_mean = float(np.mean(samples))
    emp_var = float(np.var(samples, ddof=1))
    emp_std = float(np.std(samples, ddof=1))
    theo_std = float(np.sqrt(theo_var))
    
    mean_err = abs(emp_mean - theo_mean)
    var_err = abs(emp_var - theo_var)
    std_err_mean = theo_std / np.sqrt(n_samples)
    
    summary_df = pd.DataFrame([
        {"Metric": "Mean", "Theoretical": round(theo_mean, 5), "Simulated": round(emp_mean, 5), "Abs Error": round(mean_err, 5)},
        {"Metric": "Variance", "Theoretical": round(theo_var, 5), "Simulated": round(emp_var, 5), "Abs Error": round(var_err, 5)},
        {"Metric": "Std Dev", "Theoretical": round(theo_std, 5), "Simulated": round(emp_std, 5), "Abs Error": round(abs(emp_std - theo_std), 5)},
    ])
    
    return {
        "distribution": distribution_name,
        "parameters": parameters,
        "n_samples": n_samples,
        "summary_table": summary_df,
        "standard_error_mean": round(std_err_mean, 5),
        "within_2se": bool(mean_err <= 2 * std_err_mean)
    }

def main():
    print("=" * 70)
    print("  CHALLENGE 5: MONTE CARLO DISTRIBUTION ANALYZER")
    print("=" * 70)
    
    res_norm = compare_theory_vs_simulation("normal", {"mu": 100.0, "sigma": 15.0}, n_samples=50_000)
    print("\n[Test 1: Normal Distribution (mu=100, sigma=15)]")
    print(res_norm["summary_table"].to_string(index=False))
    print(f"Mean Error <= 2*SE ({res_norm['standard_error_mean'] * 2:.4f}): {res_norm['within_2se']}")
    
    res_pois = compare_theory_vs_simulation("poisson", {"lambda": 7.5}, n_samples=50_000)
    print("\n[Test 2: Poisson Distribution (lambda=7.5)]")
    print(res_pois["summary_table"].to_string(index=False))
    print(f"Mean Error <= 2*SE ({res_pois['standard_error_mean'] * 2:.4f}): {res_pois['within_2se']}")

if __name__ == "__main__":
    main()
