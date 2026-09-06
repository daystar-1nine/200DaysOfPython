"""
Stochastic simulation and empirical comparison engine.
"""
from typing import Any
import numpy as np
import pandas as pd
from distributions.base import BaseDistribution

def run_distribution_simulation(dist: BaseDistribution, n_samples: int = 50_000, seed: int = 42) -> dict[str, Any]:
    """
    Generate Monte Carlo samples from the distribution and compute empirical moments,
    percentiles, and absolute / relative deviations from theoretical parameters.
    """
    samples = dist.sample(size=n_samples, random_state=seed)
    
    emp_mean = float(np.mean(samples))
    emp_var = float(np.var(samples, ddof=1))
    emp_std = float(np.std(samples, ddof=1))
    emp_median = float(np.median(samples))
    emp_p10 = float(np.percentile(samples, 10))
    emp_p90 = float(np.percentile(samples, 90))
    
    theo_mean = dist.mean()
    theo_var = dist.variance()
    theo_std = dist.std()
    theo_median = float(dist.ppf(0.50))
    theo_p10 = float(dist.ppf(0.10))
    theo_p90 = float(dist.ppf(0.90))
    
    mean_err = abs(emp_mean - theo_mean)
    var_err = abs(emp_var - theo_var)
    std_err = abs(emp_std - theo_std)
    se_of_mean = theo_std / np.sqrt(n_samples) if n_samples > 0 else 0.0
    
    metrics = [
        {"Metric": "Mean", "Theoretical": round(theo_mean, 5), "Simulated": round(emp_mean, 5), "Abs_Error": round(mean_err, 5)},
        {"Metric": "Variance", "Theoretical": round(theo_var, 5), "Simulated": round(emp_var, 5), "Abs_Error": round(var_err, 5)},
        {"Metric": "Std Dev", "Theoretical": round(theo_std, 5), "Simulated": round(emp_std, 5), "Abs_Error": round(std_err, 5)},
        {"Metric": "Median", "Theoretical": round(theo_median, 5), "Simulated": round(emp_median, 5), "Abs_Error": round(abs(emp_median - theo_median), 5)},
        {"Metric": "P10", "Theoretical": round(theo_p10, 5), "Simulated": round(emp_p10, 5), "Abs_Error": round(abs(emp_p10 - theo_p10), 5)},
        {"Metric": "P90", "Theoretical": round(theo_p90, 5), "Simulated": round(emp_p90, 5), "Abs_Error": round(abs(emp_p90 - theo_p90), 5)},
    ]
    df_metrics = pd.DataFrame(metrics)
    df_metrics["Distribution"] = dist.name
    
    return {
        "distribution": dist.name,
        "n_samples": n_samples,
        "samples": samples,
        "metrics_df": df_metrics,
        "se_of_mean": round(se_of_mean, 6),
        "mean_converged_2se": bool(mean_err <= 2 * se_of_mean)
    }
