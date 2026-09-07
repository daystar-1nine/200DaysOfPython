"""
Executive ASCII report and CSV exporter.
"""
from pathlib import Path
import pandas as pd

class ReportGenerator:
    def __init__(self, report_path: Path):
        self.report_path = Path(report_path)

    def generate(
        self,
        descriptive_stats: dict,
        t_res: dict,
        levels_dict: dict,
        prop_res: dict,
        sample_size_res: dict,
        boot_res: dict,
        insights: list[str]
    ) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("  DAY 69: CONFIDENCE INTERVAL & POPULATION ESTIMATION REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append("[1. SAMPLE DESCRIPTIVE STATISTICS]")
        lines.append("-" * 80)
        lines.append(f"Sample Size (n):              {descriptive_stats['count']:,}")
        lines.append(f"Sample Mean (Point Estimate): Rs. {descriptive_stats['mean']:,.2f}")
        lines.append(f"Sample Median:                Rs. {descriptive_stats['median']:,.2f}")
        lines.append(f"Sample Standard Deviation:    Rs. {descriptive_stats['std_dev']:,.2f}")
        lines.append(f"Standard Error (SE):          Rs. {descriptive_stats['standard_error']:,.2f}")
        lines.append(f"Interquartile Range (IQR):    Rs. {descriptive_stats['iqr']:,.2f} (Q1: {descriptive_stats['q1']:,.2f}, Q3: {descriptive_stats['q3']:,.2f})")
        lines.append(f"Range [Min, Max]:             Rs. [{descriptive_stats['min']:,.2f}, {descriptive_stats['max']:,.2f}]")
        lines.append("")
        lines.append("[2. 95% CONFIDENCE INTERVAL (STUDENT'S T-DISTRIBUTION)]")
        lines.append("-" * 80)
        lines.append(f"Estimation Method:            {t_res['method']}")
        lines.append(f"Degrees of Freedom (df):      {t_res['degrees_of_freedom']}")
        lines.append(f"Critical t-Value:             {t_res['critical_value']:.4f}")
        lines.append(f"Margin of Error (ME):         +/- Rs. {t_res['margin_of_error']:,.2f}")
        lines.append(f"95% Confidence Interval:      Rs. [{t_res['lower_bound']:,.2f}, {t_res['upper_bound']:,.2f}]")
        lines.append(f"Interval Width:               Rs. {t_res['interval_width']:,.2f}")
        lines.append("")
        lines.append("[3. MULTI-LEVEL CONFIDENCE COMPARISON]")
        lines.append("-" * 80)
        lines.append(f"{'Level':<8} {'Critical Value':<16} {'Margin of Error':<18} {'Interval Span':<24} {'Width':<10}")
        lines.append("-" * 80)
        for lvl, d in levels_dict.items():
            span = f"Rs. [{d['lower_bound']:,.2f}, {d['upper_bound']:,.2f}]"
            lines.append(f"{lvl:<8} {d['critical_value']:<16.4f} +/- Rs. {d['margin_of_error']:<12,.2f} {span:<24} Rs. {d['interval_width']:,.2f}")
        lines.append("")
        lines.append("[4. PROPORTION ESTIMATION (REPEAT CUSTOMER RATE)]")
        lines.append("-" * 80)
        lines.append(f"Survey / Sample Count:        {prop_res['successes']} / {prop_res['sample_size']} ({prop_res['sample_proportion']:.1%})")
        lines.append(f"Estimation Method:            {prop_res['method']}")
        lines.append(f"Standard Error:               {prop_res['standard_error']:.4f}")
        lines.append(f"Margin of Error:              +/- {prop_res['margin_of_error']:.4f}")
        lines.append(f"95% Confidence Interval:      [{prop_res['lower_bound']:.4f}, {prop_res['upper_bound']:.4f}] ({prop_res['lower_bound']*100:.1f}% to {prop_res['upper_bound']*100:.1f}%)")
        lines.append("")
        lines.append("[5. SAMPLE SIZE PLANNING MATRIX]")
        lines.append("-" * 80)
        for target_e, n_req in sample_size_res.items():
            lines.append(f"Target ME: +/- Rs. {target_e:<6} -> Required Minimum Sample Size: {n_req:,} orders")
        lines.append("")
        lines.append("[6. NON-PARAMETRIC BOOTSTRAP ESTIMATION (10,000 ITERATIONS)]")
        lines.append("-" * 80)
        lines.append(f"Observed Sample Mean:         Rs. {boot_res['observed_statistic']:,.2f}")
        lines.append(f"Bootstrap Distribution Mean:  Rs. {boot_res['bootstrap_mean']:,.2f}")
        lines.append(f"Bootstrap Standard Error:     Rs. {boot_res['bootstrap_se']:,.2f}")
        lines.append(f"95% Percentile Interval:      Rs. [{boot_res['lower_bound']:,.2f}, {boot_res['upper_bound']:,.2f}]")
        lines.append("")
        lines.append("[7. EXECUTIVE BUSINESS TAKEAWAYS & INTERPRETATION]")
        lines.append("-" * 80)
        for ins in insights:
            lines.append(ins)
        lines.append("=" * 80)

        report_text = "\n".join(lines)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        return report_text
