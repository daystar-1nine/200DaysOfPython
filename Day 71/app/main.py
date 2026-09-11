"""
Main command-line entry point for Day 71 A/B Testing Analysis Engine.
"""

import os
import pandas as pd

try:
    from app.config import config
    from app.validator import validate_groups
    from app.metrics import (
        compute_conversion_metrics,
        compute_financial_metrics,
        compute_guardrail_metrics,
        check_sample_ratio_mismatch
    )
    from app.proportion_tests import two_proportion_ztest
    from app.confidence_intervals import proportion_difference_ci
    from app.business_impact import project_annual_business_impact
    from app.decision import evaluate_ab_test_decision
    from app.visualizations import Visualizer
    from app.report import ReportGenerator
except (ImportError, ModuleNotFoundError):
    from config import config
    from validator import validate_groups
    from metrics import (
        compute_conversion_metrics,
        compute_financial_metrics,
        compute_guardrail_metrics,
        check_sample_ratio_mismatch
    )
    from proportion_tests import two_proportion_ztest
    from confidence_intervals import proportion_difference_ci
    from business_impact import project_annual_business_impact
    from decision import evaluate_ab_test_decision
    from visualizations import Visualizer
    from report import ReportGenerator

def main():
    print("=" * 75)
    print("Day 71: A/B Testing Analysis Engine")
    print("=" * 75)
    
    # 1. Load Data
    print("1. Ingesting experiment dataset...")
    df_users = pd.read_csv(config.USERS_CSV)
    ctrl, trt = validate_groups(df_users)
    print(f"   * Loaded {len(df_users):,} users ({len(ctrl):,} Control, {len(trt):,} Treatment)")
    
    # 2. Check Sample Ratio Mismatch (SRM)
    srm_res = check_sample_ratio_mismatch(len(ctrl), len(trt))
    print(f"   * SRM Check: Chi2={srm_res['chi2_statistic']:.4f}, p={srm_res['p_value']:.4f} (SRM Detected: {srm_res['srm_detected']})")
    
    # 3. Compute Metrics
    print("\n2. Evaluating Experiment Metrics...")
    conv = compute_conversion_metrics(ctrl, trt)
    fin = compute_financial_metrics(ctrl, trt)
    guard = compute_guardrail_metrics(ctrl, trt)
    
    print(f"   * Control Conversion   : {conv['rate_control']*100:.2f}% ({conv['conversions_control']:,} conv)")
    print(f"   * Treatment Conversion : {conv['rate_treatment']*100:.2f}% ({conv['conversions_treatment']:,} conv)")
    print(f"   * Absolute Lift        : {conv['absolute_lift']*100:+.2f} percentage points")
    print(f"   * Relative Lift        : {conv['relative_lift_pct']:+.2f}%")
    print(f"   * Control ARPU         : ${fin['arpu_control']:.2f}")
    print(f"   * Treatment ARPU       : ${fin['arpu_treatment']:.2f} (Lift: ${fin['arpu_lift']:+.2f})")
    
    # 4. Statistical Testing & Confidence Interval
    print("\n3. Conducting Two-Proportion Pooled Z-Test & CI Estimation...")
    stats_res = two_proportion_ztest(
        success_c=conv["conversions_control"],
        total_c=conv["n_control"],
        success_t=conv["conversions_treatment"],
        total_t=conv["n_treatment"],
        alpha=config.DEFAULT_ALPHA,
        alternative="two-sided"
    )
    ci_res = proportion_difference_ci(
        p_control=conv["rate_control"],
        n_control=conv["n_control"],
        p_treatment=conv["rate_treatment"],
        n_treatment=conv["n_treatment"],
        confidence=0.95
    )
    stats_res["ci_lower"] = ci_res["ci_lower"]
    stats_res["ci_upper"] = ci_res["ci_upper"]
    
    print(f"   * z-statistic          : {stats_res['z_statistic']:.4f}")
    print(f"   * p-value              : {stats_res['p_value']:.6e}")
    print(f"   * 95% Conf Interval    : [{ci_res['ci_lower']*100:+.2f}%, {ci_res['ci_upper']*100:+.2f}%]")
    print(f"   * Statistical Reject   : {stats_res['reject_null']}")
    
    # 5. Financial Sizing & Decision
    print("\n4. Sizing Business Impact & Formulating Launch Recommendation...")
    impact = project_annual_business_impact(
        absolute_lift=conv["absolute_lift"],
        aov=fin["aov_treatment"] if fin["aov_treatment"] > 0 else 50.0,
        monthly_visitors=500000,
        cost_to_deploy=25000.0
    )
    decision = evaluate_ab_test_decision(
        p_value=stats_res["p_value"],
        alpha=config.DEFAULT_ALPHA,
        relative_lift_pct=conv["relative_lift_pct"],
        mde_pct=8.0,
        bounce_delta=guard["bounce_delta"],
        max_allowed_bounce_delta=0.05,
        refund_delta=guard["refund_delta"],
        max_allowed_refund_delta=0.015,
        srm_detected=srm_res["srm_detected"]
    )
    print(f"   * Projected Annual Net Gain : ${impact['projected_net_annual_profit']:,.2f}")
    print(f"   * Decision                  : {decision.launch_recommendation}")
    print(f"   * Rationale                 : {decision.rationale}")

    # 6. Generate Visualizations
    print("\n5. Rendering 8 Publication-Grade Visualizations...")
    viz = Visualizer()
    charts = viz.generate_all(df_users, conv, stats_res, decision, impact)
    for c in charts:
        print(f"   * Created chart: {c}")
        
    # 7. Export Reports
    print("\n6. Exporting Comprehensive Reports...")
    rep = ReportGenerator()
    rep.export_experiment_summary(conv, fin, guard)
    rep.export_statistical_results(stats_res)
    rep.export_business_impact_report(impact)
    rep.export_full_text_report(conv, fin, stats_res, guard, decision, impact)
    print(f"   * Experiment Summary CSV : {config.EXP_SUMMARY_CSV}")
    print(f"   * Statistical Results CSV: {config.STATS_RESULTS_CSV}")
    print(f"   * Business Impact TXT    : {config.BUSINESS_IMPACT_TXT}")
    print(f"   * Full Executive Report  : {config.REPORT_TXT}")
    
    print("\nA/B Testing Engine execution completed successfully!")

if __name__ == "__main__":
    main()
