"""
Main entry point for Day 66 Probability Simulator.
Executes the analytical pipeline, generates 5 publication charts,
exports 4 CSV datasets, and saves the final executive report.
"""
import sys
from pathlib import Path

# Add Day 66 folder to sys.path
day66_dir = Path(__file__).resolve().parent.parent
if str(day66_dir) not in sys.path:
    sys.path.insert(0, str(day66_dir))

from app.analysis import run_full_pipeline
from app.visualizations import (
    plot_coin_convergence,
    plot_dice_distribution,
    plot_dice_sum_distribution,
    plot_expected_vs_actual,
    plot_fraud_probability
)
from app.report import export_all_datasets, generate_ascii_report

def main():
    print("=" * 78)
    print("   DAY 66: REAL-WORLD PROBABILITY SIMULATION ENGINE")
    print("=" * 78)
    
    # 1. Run Pipeline
    results = run_full_pipeline()
    
    # 2. Generate Visualizations
    print("\n--> Generating Publication Visualizations...")
    plot_coin_convergence(results["coin"])
    plot_dice_distribution(results["dice"])
    plot_dice_sum_distribution(results["dice_two"])
    plot_expected_vs_actual(results["risk"])
    plot_fraud_probability(results["fraud_bayes"])
    
    # 3. Export Datasets
    print("\n--> Exporting Simulation Datasets...")
    export_all_datasets(results)
    
    # 4. Generate ASCII Report
    print("\n--> Generating Analytical Report...")
    report_str = generate_ascii_report(results)
    
    print("\n[SUCCESS] Day 66 Pipeline completed with all artifacts generated cleanly.")

if __name__ == "__main__":
    main()
