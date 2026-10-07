"""
Generates all 12 publication charts and final markdown evaluation report.
"""
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
day116_dir = current_dir.parent
if str(day116_dir) not in sys.path:
    sys.path.insert(0, str(day116_dir))

from app.reporting.charts import generate_all_charts
from app.reporting.report import generate_final_evaluation_report


def main():
    print("=" * 70)
    print("  [MiniGPT Evaluation Harness] - Report and Visuals Generator")
    print("=" * 70)

    outputs_dir = day116_dir / "outputs"
    metrics_dir = outputs_dir / "metrics"
    raw_dir = outputs_dir / "raw"
    reports_dir = outputs_dir / "reports"
    charts_dir = outputs_dir / "charts"

    print("\nGenerating 12 publication-grade evaluation charts...")
    generated_charts = generate_all_charts(metrics_dir, raw_dir, reports_dir, charts_dir)
    print(f"Successfully generated {len(generated_charts)} charts in: {charts_dir}")

    report_path = reports_dir / "final_evaluation.md"
    print(f"\nCompiling executive evaluation markdown report to: {report_path}")
    generate_final_evaluation_report(metrics_dir, raw_dir, reports_dir, report_path)
    print("Report compilation complete!")


if __name__ == "__main__":
    main()
