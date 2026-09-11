"""
Report generator for statistical hypothesis test outputs.
"""

import os
import pandas as pd

try:
    from app.config import config
    from app.hypotheses import TestResult
except (ImportError, ModuleNotFoundError):
    from config import config
    from hypotheses import TestResult

class ReportGenerator:
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or config.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def export_results_csv(self, results: list[TestResult], output_path: str = None) -> str:
        path = output_path or config.RESULTS_CSV
        rows = []
        for r in results:
            rows.append({
                "test_id": r.spec.test_id,
                "scenario_name": r.spec.scenario_name,
                "parameter": r.spec.parameter_name,
                "test_type": r.spec.test_kind.value,
                "null_value": r.spec.null_value,
                "alternative": r.spec.alternative.value,
                "alpha": r.spec.alpha,
                "sample_size": r.sample_size,
                "point_estimate": r.point_estimate,
                "standard_error": r.standard_error,
                "test_statistic": r.test_statistic,
                "critical_value": r.critical_value,
                "p_value": r.p_value,
                "ci_lower": r.ci_lower,
                "ci_upper": r.ci_upper,
                "cohens_d": r.cohens_d if r.cohens_d is not None else "N/A",
                "effect_magnitude": r.effect_magnitude,
                "reject_null": r.reject_null,
                "decision": r.decision_pvalue,
                "summary": r.summary
            })
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        return path

    def export_text_report(self, results: list[TestResult], output_path: str = None) -> str:
        path = output_path or config.REPORT_TXT
        lines = []
        lines.append("=" * 80)
        lines.append("STATISTICAL HYPOTHESIS TESTING ENGINE — EXECUTIVE SUMMARY REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        for idx, r in enumerate(results, 1):
            lines.append(f"Scenario {idx}: {r.spec.scenario_name}")
            lines.append("-" * 75)
            lines.append(f"  * Test ID           : {r.spec.test_id}")
            lines.append(f"  * Test Kind         : {r.spec.test_kind.value}")
            lines.append(f"  * Null Claim (H0)   : {r.spec.parameter_name} = {r.spec.null_value}")
            lines.append(f"  * Alternative (H1)  : {r.spec.parameter_name} ({r.spec.alternative.value}) {r.spec.null_value}")
            lines.append(f"  * Sample Size (n)   : {r.sample_size}")
            lines.append(f"  * Point Estimate    : {r.point_estimate:.4f}")
            lines.append(f"  * Standard Error    : {r.standard_error:.4f}")
            lines.append(f"  * Test Statistic    : {r.test_statistic:.4f}")
            lines.append(f"  * Critical Value    : {r.critical_value:.4f}")
            lines.append(f"  * P-Value           : {r.p_value:.6e}")
            lines.append(f"  * 95% Conf Interval : [{r.ci_lower:.4f}, {r.ci_upper:.4f}]")
            if r.cohens_d is not None:
                lines.append(f"  * Cohen's d (Effect): {r.cohens_d:.4f} ({r.effect_magnitude})")
            lines.append(f"  * Rejection Decision: {'REJECT NULL HYPOTHESIS' if r.reject_null else 'FAIL TO REJECT NULL'}")
            lines.append(f"  * Business Finding  : {r.summary}")
            lines.append("")
            
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        content = "\n".join(lines)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
