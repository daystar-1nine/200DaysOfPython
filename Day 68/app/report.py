"""
Report generator for Day 68 simulation results.
"""
from pathlib import Path
import pandas as pd

class ReportGenerator:
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)

    def generate(self, clt_df: pd.DataFrame, boot_res: dict, sample_df: pd.DataFrame | None = None) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("  DAY 68: SAMPLING, SAMPLING DISTRIBUTIONS & CLT INFERENCE REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append("[1. CENTRAL LIMIT THEOREM CONVERGENCE SUMMARY]")
        lines.append("-" * 80)
        lines.append(clt_df.to_string(index=False))
        lines.append("")
        lines.append("[2. BOOTSTRAP RESAMPLING ANALYSIS (SINGLE SAMPLE n=50)]")
        lines.append("-" * 80)
        lines.append(f"Sample Observed Mean:       {boot_res['observed']:.4f}")
        lines.append(f"Bootstrap Distribution Mean: {boot_res['boot_mean']:.4f}")
        lines.append(f"Bootstrap Standard Error:   {boot_res['boot_se']:.4f}")
        lines.append(f"{boot_res['ci_level']*100:.0f}% Percentile CI:            [{boot_res['ci_lower']:.4f}, {boot_res['ci_upper']:.4f}]")
        if 'true_population_mean' in boot_res:
            lines.append(f"True Population Mean:       {boot_res['true_population_mean']:.4f}")
            captured = boot_res['ci_lower'] <= boot_res['true_population_mean'] <= boot_res['ci_upper']
            lines.append(f"Confidence Interval Holds:  {captured}")
        lines.append("")
        lines.append("[3. KEY STATISTICAL TAKEAWAYS]")
        lines.append("-" * 80)
        lines.append("1. Unbiasedness: The mean of the sampling distribution perfectly tracks the population mean.")
        lines.append("2. Standard Error Decay: Uncertainty decreases by 1/sqrt(n), exhibiting diminishing returns.")
        lines.append("3. Central Limit Theorem: Skewed population sample means achieve Gaussian bell shape as n >= 30.")
        lines.append("4. Bootstrap Robustness: Allows confidence interval construction without knowing population sigma.")
        lines.append("=" * 80)

        report_text = "\n".join(lines)
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        return report_text
