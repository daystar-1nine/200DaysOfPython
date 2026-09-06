"""
Main entry point for Day 68 Capstone Simulator.
"""
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from app.config import SimulationConfig
    from app.analyzer import SamplingAnalyzer
    from app.visualizations import Visualizer
    from app.report import ReportGenerator
except (ImportError, ModuleNotFoundError):
    from config import SimulationConfig
    from analyzer import SamplingAnalyzer
    from visualizations import Visualizer
    from report import ReportGenerator


def main():
    print("=================================================================")
    print("  DAY 68: SAMPLING & CENTRAL LIMIT THEOREM SIMULATOR")
    print("=================================================================")
    config = SimulationConfig()
    analyzer = SamplingAnalyzer(config)
    viz = Visualizer(config.charts_dir)

    print("-> 1. Generating Population Visualizations...")
    viz.plot_population_distributions(analyzer.populations, "population.png")

    print("-> 2. Evaluating Single Sample vs Population...")
    pop_skew = analyzer.populations["Exponential (Skewed)"]
    single_sample = pop_skew[:config.default_sample_size]
    viz.plot_sample_vs_population(pop_skew, single_sample, "sample.png")

    print("-> 3. Running CLT Convergence Experiment...")
    clt_df = analyzer.run_clt_analysis(pop_name="Exponential (Skewed)")
    print(clt_df.to_string(index=False))

    # Export sampling results
    clt_df.to_csv(config.sampling_results_file, index=False)
    print(f"   Saved sampling results to: {config.sampling_results_file.name}")

    print("-> 4. Generating Sampling Distribution Charts...")
    viz.plot_sampling_distribution(pop_skew, config.sample_sizes, n_samples=config.n_simulations, filename="sampling_distribution.png")
    viz.plot_clt_comparison(pop_skew, list(config.sample_sizes), n_samples=config.n_simulations, filename="clt_comparison.png")

    print("-> 5. Plotting Standard Error Inverse-Square Curve...")
    pop_std = float(np.std(pop_skew))
    sample_sizes_list = list(clt_df["sample_size"].values)
    emp_ses_list = list(clt_df["emp_se"].values)
    viz.plot_standard_error(pop_std, sample_sizes_list, emp_ses_list, filename="standard_error.png")

    print("-> 6. Generating Theoretical vs Experimental Distribution Charts...")
    # Using n = 30 simulation
    sim_n30 = analyzer.run_clt_analysis(pop_name="Exponential (Skewed)")
    means_n30 = np.array([np.mean(np.random.default_rng(config.random_seed).choice(pop_skew, size=30, replace=False)) for _ in range(config.n_simulations)])
    theo_se_n30 = pop_std / np.sqrt(30)
    viz.plot_theoretical_vs_experimental(means_n30, float(np.mean(pop_skew)), theo_se_n30, filename="theoretical_vs_experimental.png")

    print("-> 7. Executing Bootstrap Resampling Analysis...")
    boot_res = analyzer.run_bootstrap_analysis(pop_name="Exponential (Skewed)", sample_size=50)
    viz.plot_bootstrap_ci(boot_res, filename="bootstrap.png")

    # Export bootstrap distribution
    df_boot = pd.DataFrame({"bootstrap_means": boot_res["distribution"]})
    df_boot.to_csv(config.bootstrap_results_file, index=False)
    print(f"   Saved bootstrap results to: {config.bootstrap_results_file.name}")

    print("-> 8. Generating Final Statistical Report...")
    reporter = ReportGenerator(config.report_file)
    rep_text = reporter.generate(clt_df, boot_res)
    print(f"   Saved report to: {config.report_file.name}")
    print("=================================================================")
    print("  SIMULATION COMPLETE! 7 CHARTS GENERATED IN output/charts/")
    print("=================================================================")

if __name__ == "__main__":
    main()
