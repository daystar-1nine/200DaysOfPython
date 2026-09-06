"""
Main entry point for Day 67 Probability Distribution Analyzer.
Executes the analytical pipeline, generates 8 publication charts,
exports 3 CSV datasets, and saves the final executive report.
"""
import sys
from pathlib import Path

day67_dir = Path(__file__).resolve().parent.parent
if str(day67_dir) not in sys.path:
    sys.path.insert(0, str(day67_dir))

import pandas as pd
from app.config import DEFAULT_DISTRIBUTIONS, DEFAULT_SAMPLE_SIZE, RANDOM_SEED
from app.analyzer import create_distribution, evaluate_distribution_probabilities
from app.simulator import run_distribution_simulation
from app.visualizations import (
    plot_bernoulli,
    plot_binomial,
    plot_uniform,
    plot_normal,
    plot_poisson,
    plot_normal_comparison,
    plot_binomial_simulation,
    plot_normal_simulation
)
from app.report import export_all_datasets, generate_ascii_report

def main():
    print("=" * 78)
    print("   DAY 67: PROBABILITY DISTRIBUTION ANALYZER & SIMULATOR")
    print("=" * 78)
    
    dist_instances = {}
    summary_records = []
    all_probs = []
    all_sims = []
    
    # 1. Theoretical Analysis & Moments
    print("\n--> [1/4] Evaluating Theoretical Distributions...")
    for name, params in DEFAULT_DISTRIBUTIONS.items():
        dist = create_distribution(name, params)
        dist_instances[name] = dist
        
        summary_records.append({
            "Distribution": dist.name,
            "Parameters": str(dist.parameters),
            "Mean": round(dist.mean(), 4),
            "Variance": round(dist.variance(), 4),
            "Std_Dev": round(dist.std(), 4),
            "Is_Discrete": dist.is_discrete
        })
        
        df_p = evaluate_distribution_probabilities(dist)
        all_probs.append(df_p)
        
    df_summary = pd.DataFrame(summary_records)
    df_all_probs = pd.concat(all_probs, ignore_index=True)
    
    # 2. Stochastic Simulations (Theory vs Simulation)
    print("--> [2/4] Executing Monte Carlo Simulations (N = 50,000)...")
    for name, dist in dist_instances.items():
        sim_res = run_distribution_simulation(dist, n_samples=DEFAULT_SAMPLE_SIZE, seed=RANDOM_SEED)
        all_sims.append(sim_res["metrics_df"])
        
    df_all_sims = pd.concat(all_sims, ignore_index=True)
    
    # 3. Generate Visualizations (8 charts)
    print("--> [3/4] Generating 8 Publication-Grade Visualizations...")
    plot_bernoulli(p=0.7)
    plot_binomial(n=20, p=0.4)
    plot_uniform(a=0.0, b=10.0)
    plot_normal(mu=70.0, sigma=10.0)
    plot_poisson(lam=5.0)
    plot_normal_comparison(mu=50.0, sigmas=[5.0, 10.0, 20.0])
    plot_binomial_simulation(n=20, p=0.5, n_sims=10_000)
    plot_normal_simulation(mu=70.0, sigma=10.0, n_sims=10_000)
    
    # 4. Export Datasets & ASCII Report
    print("--> [4/4] Exporting Datasets & Analytical Report...")
    export_all_datasets(df_summary, df_all_probs, df_all_sims)
    generate_ascii_report(df_summary, df_all_probs, df_all_sims)
    
    print("\n[SUCCESS] Day 67 Distribution Engine completed with all artifacts generated cleanly.")

if __name__ == "__main__":
    main()
