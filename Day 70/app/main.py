"""
Main command-line entry point for Day 70 Hypothesis Testing Engine.
"""

import sys

try:
    from app.scenarios import run_all_scenarios
    from app.visualizations import Visualizer
    from app.report import ReportGenerator
except (ImportError, ModuleNotFoundError):
    from scenarios import run_all_scenarios
    from visualizations import Visualizer
    from report import ReportGenerator

def main():
    print("=" * 75)
    print("Day 70: Statistical Hypothesis Testing Engine")
    print("=" * 75)
    print("1. Running 4 Business Hypothesis Testing Scenarios...")
    results = run_all_scenarios()
    
    for r in results:
        status = "REJECT H0" if r.reject_null else "FAIL TO REJECT"
        print(f"   [{status:14s}] {r.spec.scenario_name:<42s} | p={r.p_value:.4e} | Stat={r.test_statistic:+.2f}")

    print("\n2. Generating 7 Publication-Grade Visualizations...")
    viz = Visualizer()
    chart_paths = viz.generate_all_charts(results)
    for p in chart_paths:
        print(f"   * Created chart: {p}")

    print("\n3. Exporting Analysis Reports...")
    rep = ReportGenerator()
    csv_path = rep.export_results_csv(results)
    txt_path = rep.export_text_report(results)
    print(f"   * Results CSV: {csv_path}")
    print(f"   * Summary TXT: {txt_path}")
    
    print("\nExecution completed successfully!")

if __name__ == "__main__":
    main()
