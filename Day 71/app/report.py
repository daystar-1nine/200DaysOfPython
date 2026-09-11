"""
Report generation and export module for A/B Testing Analysis Engine.
"""

import os
import pandas as pd

try:
    from app.config import config
except (ImportError, ModuleNotFoundError):
    from config import config

class ReportGenerator:
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or config.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def export_experiment_summary(self, conv: dict, fin: dict, guard: dict, path: str = None) -> str:
        filepath = path or config.EXP_SUMMARY_CSV
        rows = [
            {"category": "Sample", "metric": "Control Users", "value": conv["n_control"]},
            {"category": "Sample", "metric": "Treatment Users", "value": conv["n_treatment"]},
            {"category": "Conversion", "metric": "Control Rate", "value": f"{conv['rate_control']*100:.2f}%"},
            {"category": "Conversion", "metric": "Treatment Rate", "value": f"{conv['rate_treatment']*100:.2f}%"},
            {"category": "Conversion", "metric": "Absolute Lift", "value": f"{conv['absolute_lift']*100:+.2f} pp"},
            {"category": "Conversion", "metric": "Relative Lift", "value": f"{conv['relative_lift_pct']:+.2f}%"},
            {"category": "Financial", "metric": "Control ARPU", "value": f"${fin['arpu_control']:.2f}"},
            {"category": "Financial", "metric": "Treatment ARPU", "value": f"${fin['arpu_treatment']:.2f}"},
            {"category": "Financial", "metric": "ARPU Lift", "value": f"${fin['arpu_lift']:+.2f}"},
            {"category": "Guardrail", "metric": "Bounce Delta", "value": f"{guard['bounce_delta']:+.2%}"},
            {"category": "Guardrail", "metric": "Refund Delta", "value": f"{guard['refund_delta']:+.2%}"}
        ]
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        return filepath

    def export_statistical_results(self, stats_res: dict, path: str = None) -> str:
        filepath = path or config.STATS_RESULTS_CSV
        rows = [{
            "test_type": stats_res["test_type"],
            "statistic_name": "z" if "z_statistic" in stats_res else "t",
            "statistic_value": stats_res.get("z_statistic", stats_res.get("t_statistic")),
            "critical_value": stats_res["critical_value"],
            "p_value": stats_res["p_value"],
            "alpha": stats_res["alpha"],
            "ci_lower": stats_res.get("ci_lower", "N/A"),
            "ci_upper": stats_res.get("ci_upper", "N/A"),
            "reject_null": stats_res["reject_null"]
        }]
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        return filepath

    def export_business_impact_report(self, impact: dict, path: str = None) -> str:
        filepath = path or config.BUSINESS_IMPACT_TXT
        lines = [
            "=" * 65,
            "A/B TEST BUSINESS IMPACT PROJECTIONS",
            "=" * 65,
            f"Monthly Unique Visitors            : {impact['monthly_visitors']:,}",
            f"Annual Unique Visitors             : {impact['annual_visitors']:,}",
            f"Absolute Conversion Lift           : {impact['absolute_conversion_lift']*100:+.2f} percentage points",
            f"Average Order Value (AOV)          : ${impact['average_order_value']:.2f}",
            "-" * 65,
            f"Annual Incremental Conversions     : {impact['annual_incremental_conversions']:,} orders",
            f"Projected Annual Gross Revenue     : ${impact['projected_annual_gross_revenue']:,.2f}",
            f"Deployment & Engineering Cost      : ${impact['deployment_cost']:,.2f}",
            f"Projected Annual Net Profit        : ${impact['projected_net_annual_profit']:,.2f}",
            f"Projected ROI                      : {impact['projected_roi_pct']:,.1f}%",
            "=" * 65
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return filepath

    def export_full_text_report(self, conv: dict, fin: dict, stats_res: dict, guard: dict, decision, impact: dict, path: str = None) -> str:
        filepath = path or config.REPORT_TXT
        lines = [
            "=" * 70,
            "A/B TESTING ANALYSIS ENGINE — EXECUTIVE EXPERIMENTATION REPORT",
            "=" * 70,
            "",
            "1. SAMPLE & TRAFFIC ALLOCATION",
            "-" * 65,
            f"  Control Users    : {conv['n_control']:,}",
            f"  Treatment Users  : {conv['n_treatment']:,}",
            f"  Total Sample Size: {conv['n_control'] + conv['n_treatment']:,}",
            "",
            "2. PRIMARY METRIC: CONVERSION RATE",
            "-" * 65,
            f"  Control Conversion   : {conv['conversions_control']:,} / {conv['n_control']:,} = {conv['rate_control']*100:.2f}%",
            f"  Treatment Conversion : {conv['conversions_treatment']:,} / {conv['n_treatment']:,} = {conv['rate_treatment']*100:.2f}%",
            f"  Absolute Lift        : {conv['absolute_lift']*100:+.2f} percentage points",
            f"  Relative Lift        : {conv['relative_lift_pct']:+.2f}%",
            "",
            "3. STATISTICAL TEST RESULTS",
            "-" * 65,
            f"  Test Performed       : {stats_res['test_type']}",
            f"  Test Statistic       : {stats_res.get('z_statistic', stats_res.get('t_statistic')):.4f}",
            f"  Critical Threshold   : {stats_res['critical_value']:.4f}",
            f"  P-Value              : {stats_res['p_value']:.6e}",
            f"  Significance Level   : alpha = {stats_res['alpha']}",
            f"  Statistical Decision : {'REJECT NULL HYPOTHESIS (Uplift Verified)' if stats_res['reject_null'] else 'FAIL TO REJECT NULL'}",
            "",
            "4. FINANCIAL & REVENUE YIELD",
            "-" * 65,
            f"  Control ARPU         : ${fin['arpu_control']:.2f}",
            f"  Treatment ARPU       : ${fin['arpu_treatment']:.2f}",
            f"  ARPU Uplift          : ${fin['arpu_lift']:+.2f}",
            f"  Average Order Value  : ${fin['aov_treatment']:.2f}",
            "",
            "5. GUARDRAIL METRIC AUDIT",
            "-" * 65,
            f"  Bounce Rate Delta    : {guard['bounce_delta']:+.2%} (Control: {guard['bounce_rate_control']:.2%}, Treatment: {guard['bounce_rate_treatment']:.2%})",
            f"  Refund Rate Delta    : {guard['refund_delta']:+.2%} (Control: {guard['refund_rate_control']:.2%}, Treatment: {guard['refund_rate_treatment']:.2%})",
            f"  Session Duration     : {guard['duration_delta']:+.1f}s (Control: {guard['session_duration_control']:.1f}s, Treatment: {guard['session_duration_treatment']:.1f}s)",
            f"  Guardrails Passed    : {decision.guardrails_passed}",
            "",
            "6. BUSINESS IMPACT SIZING",
            "-" * 65,
            f"  Annual Incremental Orders : {impact['annual_incremental_conversions']:,}",
            f"  Annual Net Revenue Gain   : ${impact['projected_net_annual_profit']:,.2f}",
            f"  Projected Launch ROI      : {impact['projected_roi_pct']:,.1f}%",
            "",
            "7. FINAL EXECUTIVE DECISION & RECOMMENDATION",
            "=" * 70,
            f"  RECOMMENDATION: {decision.launch_recommendation}",
            f"  RATIONALE     : {decision.rationale}",
            "=" * 70
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return filepath
