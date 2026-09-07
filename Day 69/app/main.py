"""
Production pipeline entry point for Day 69 Capstone Analyzer.
"""
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from app.config import AppConfig
    from app.loader import DataLoader
    from app.stats_calc import compute_descriptive_stats
    from app.confidence_intervals import ConfidenceIntervalEngine
    from app.sample_size import SampleSizeCalculator
    from app.bootstrap import BootstrapEstimator
    from app.insights import InsightsGenerator
    from app.visualizations import Visualizer
    from app.report import ReportGenerator
except (ImportError, ModuleNotFoundError):
    from config import AppConfig
    from loader import DataLoader
    from stats_calc import compute_descriptive_stats
    from confidence_intervals import ConfidenceIntervalEngine
    from sample_size import SampleSizeCalculator
    from bootstrap import BootstrapEstimator
    from insights import InsightsGenerator
    from visualizations import Visualizer
    from report import ReportGenerator

def main():
    print("=================================================================")
    print("  DAY 69: CONFIDENCE INTERVAL & POPULATION ESTIMATION ANALYZER")
    print("=================================================================")
    config = AppConfig()
    
    print("-> 1. Ingesting Customer Order Transactions...")
    data = DataLoader.load_from_csv(config.data_file, column="order_value")
    df_raw = pd.read_csv(config.data_file)
    print(f"   Loaded {len(data):,} orders from {config.data_file.name}")
    
    print("-> 2. Computing Sample Descriptive Statistics...")
    stats_dict = compute_descriptive_stats(data)
    print(f"   Mean: Rs. {stats_dict['mean']} | Median: Rs. {stats_dict['median']} | SD: Rs. {stats_dict['std_dev']}")
    
    print("-> 3. Calculating Parametric T-Confidence Intervals...")
    t_res = ConfidenceIntervalEngine.t_interval_mean(data, confidence=config.default_confidence)
    print(f"   95% CI: Rs. [{t_res['lower_bound']}, {t_res['upper_bound']}] (ME: +/- Rs. {t_res['margin_of_error']})")
    
    print("-> 4. Evaluating Multi-Level Confidence Comparisons...")
    levels_dict = {}
    ci_records = []
    for conf in config.confidence_levels:
        res = ConfidenceIntervalEngine.t_interval_mean(data, confidence=conf)
        lbl = f"{int(conf * 100)}%"
        levels_dict[lbl] = res
        ci_records.append({
            "confidence_level": lbl,
            "method": res["method"],
            "sample_mean": res["sample_mean"],
            "critical_value": res["critical_value"],
            "standard_error": res["standard_error"],
            "margin_of_error": res["margin_of_error"],
            "lower_bound": res["lower_bound"],
            "upper_bound": res["upper_bound"],
            "interval_width": res["interval_width"]
        })
    df_ci = pd.DataFrame(ci_records)
    df_ci.to_csv(config.ci_results_file, index=False)
    print(f"   Saved CI summary matrix to: {config.ci_results_file.name}")
    
    print("-> 5. Estimating Customer Proportion (Repeat Buyers)...")
    repeat_count = int(df_raw["is_repeat_customer"].sum()) if "is_repeat_customer" in df_raw.columns else 640
    prop_res = ConfidenceIntervalEngine.proportion_interval(repeat_count, len(data), confidence=0.95, method="wilson")
    print(f"   Repeat Customer Rate: {prop_res['sample_proportion']:.1%} | 95% CI: [{prop_res['lower_bound']}, {prop_res['upper_bound']}]")
    
    print("-> 6. Planning Sample Size Requirements...")
    sample_size_targets = {}
    for target_me in [10.0, 25.0, 50.0, 100.0]:
        n_req = SampleSizeCalculator.for_mean(stats_dict["std_dev"], margin_of_error=target_me, confidence=0.95)
        sample_size_targets[target_me] = n_req
        
    print("-> 7. Running Non-Parametric Bootstrap Resampling (10,000 Draws)...")
    boot_engine = BootstrapEstimator(data, seed=config.random_seed)
    boot_res = boot_engine.estimate(statistic=np.mean, iterations=config.bootstrap_iterations, confidence=0.95)
    print(f"   Bootstrap Mean: Rs. {boot_res['bootstrap_mean']} | 95% CI: [{boot_res['lower_bound']}, {boot_res['upper_bound']}]")
    
    df_boot = pd.DataFrame({"bootstrap_means": boot_res["distribution"]})
    df_boot.to_csv(config.bootstrap_results_file, index=False)
    print(f"   Saved bootstrap distribution to: {config.bootstrap_results_file.name}")
    
    print("-> 8. Generating 7 Publication-Grade Visualizations...")
    viz = Visualizer(config.charts_dir)
    viz.plot_sample_distribution(data, "sample_distribution.png")
    viz.plot_confidence_interval(t_res["sample_mean"], t_res["lower_bound"], t_res["upper_bound"], "confidence_interval.png")
    viz.plot_confidence_levels(t_res["sample_mean"], levels_dict, "confidence_levels.png")
    
    # Calculate widths across sample sizes
    sim_widths = []
    for n in config.sample_sizes:
        sub_sample = data[:n]
        sub_t = ConfidenceIntervalEngine.t_interval_mean(sub_sample, confidence=0.95)
        sim_widths.append(sub_t["interval_width"])
    viz.plot_sample_size_vs_width(list(config.sample_sizes), sim_widths, "ci_width_vs_sample_size.png")
    
    viz.plot_bootstrap_distribution(boot_res["distribution"], boot_res["observed_statistic"], boot_res["lower_bound"], boot_res["upper_bound"], "bootstrap_distribution.png")
    viz.plot_bootstrap_vs_t(t_res, boot_res, "bootstrap_confidence_interval.png")
    viz.plot_parameter_estimation_dashboard(data, t_res, boot_res, "parameter_estimation.png")
    print(f"   Generated 7 charts in: {config.charts_dir.name}")
    
    print("-> 9. Compiling Executive ASCII Report & Business Insights...")
    insights = InsightsGenerator.generate_mean_insights(t_res, boot_res)
    reporter = ReportGenerator(config.report_file)
    reporter.generate(
        descriptive_stats=stats_dict,
        t_res=t_res,
        levels_dict=levels_dict,
        prop_res=prop_res,
        sample_size_res=sample_size_targets,
        boot_res=boot_res,
        insights=insights
    )
    print(f"   Saved executive report to: {config.report_file.name}")
    print("=================================================================")
    print("  SIMULATION & ESTIMATION PIPELINE COMPLETE!")
    print("=================================================================")

if __name__ == "__main__":
    main()
