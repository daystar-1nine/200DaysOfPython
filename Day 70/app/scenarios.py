"""
Execution of the 4 Real-World Business Hypothesis Testing Scenarios.
"""

try:
    from app.config import AlternativeType, TestKind
    from app.loader import DataLoader
    from app.hypotheses import HypothesisSpec, TestResult
    from app.tests_mean import one_sample_t_test
    from app.tests_proportion import one_sample_proportion_test
    from app.effect_size import compute_cohens_d
    from app.decision import evaluate_decisions
except (ImportError, ModuleNotFoundError):
    from config import AlternativeType, TestKind
    from loader import DataLoader
    from hypotheses import HypothesisSpec, TestResult
    from tests_mean import one_sample_t_test
    from tests_proportion import one_sample_proportion_test
    from effect_size import compute_cohens_d
    from decision import evaluate_decisions

def run_all_scenarios() -> list[TestResult]:
    loader = DataLoader()
    results = []
    
    # Scenario 1: Delivery Times SLA Verification
    # Claim: Delivery time <= 30 mins. Test H0: mu <= 30 vs H1: mu > 30 (Right-tailed)
    df_delivery = loader.load_delivery_times()
    data_del = df_delivery["delivery_time_minutes"].values
    spec1 = HypothesisSpec(
        test_id="SCENARIO_1_DELIVERY_SLA",
        scenario_name="E-Commerce Delivery SLA Verification",
        test_kind=TestKind.ONE_SAMPLE_T,
        parameter_name="mean_delivery_time",
        null_value=30.0,
        alternative=AlternativeType.GREATER,
        alpha=0.05,
        description="Verify whether true mean delivery time exceeds the 30.0 minute SLA guarantee."
    )
    t1 = one_sample_t_test(data_del, mu_0=30.0, alpha=0.05, alternative="greater")
    d1 = compute_cohens_d(t1["sample_mean"], 30.0, t1["sample_std"])
    dec1 = evaluate_decisions(
        p_value=t1["p_value"],
        test_statistic=t1["test_statistic"],
        critical_value=t1["critical_value"],
        alternative=AlternativeType.GREATER,
        alpha=0.05,
        ci_lower=t1["ci_lower"],
        ci_upper=t1["ci_upper"],
        null_val=30.0
    )
    res1 = TestResult(
        spec=spec1,
        sample_size=t1["n"],
        point_estimate=round(t1["sample_mean"], 4),
        standard_error=round(t1["standard_error"], 4),
        test_statistic=round(t1["test_statistic"], 4),
        critical_value=round(t1["critical_value"], 4),
        p_value=t1["p_value"],
        ci_lower=round(t1["ci_lower"], 4),
        ci_upper=round(t1["ci_upper"], 4),
        cohens_d=round(d1["cohens_d"], 4),
        effect_magnitude=d1["magnitude"],
        reject_null=dec1["reject_null"],
        decision_pvalue=dec1["decision_pvalue"],
        decision_critical=dec1["decision_critical"],
        decision_ci=dec1["decision_ci"],
        summary="Statistically significant SLA breach. Delivery times significantly exceed 30.0 minutes." if dec1["reject_null"] else "SLA compliance holds."
    )
    results.append(res1)
    
    # Scenario 2: Website Conversion Rate Optimization
    # Claim: Conversion rate > 10% (0.10). Test H0: p <= 0.10 vs H1: p > 0.10 (Right-tailed)
    df_conv = loader.load_conversion_data()
    successes = int(df_conv["converted"].sum())
    trials = len(df_conv)
    spec2 = HypothesisSpec(
        test_id="SCENARIO_2_WEBSITE_CONVERSION",
        scenario_name="Website Landing Page Conversion Optimization",
        test_kind=TestKind.PROPORTION_Z,
        parameter_name="conversion_proportion",
        null_value=0.10,
        alternative=AlternativeType.GREATER,
        alpha=0.05,
        description="Test if new onboarding funnel achieves conversion proportion strictly greater than 10% baseline."
    )
    p2 = one_sample_proportion_test(successes=successes, trials=trials, p_0=0.10, alpha=0.05, alternative="greater")
    dec2 = evaluate_decisions(
        p_value=p2["p_value"],
        test_statistic=p2["test_statistic"],
        critical_value=p2["critical_value"],
        alternative=AlternativeType.GREATER,
        alpha=0.05,
        ci_lower=p2["ci_lower"],
        ci_upper=p2["ci_upper"],
        null_val=0.10
    )
    res2 = TestResult(
        spec=spec2,
        sample_size=trials,
        point_estimate=round(p2["sample_proportion"], 4),
        standard_error=round(p2["standard_error"], 4),
        test_statistic=round(p2["test_statistic"], 4),
        critical_value=round(p2["critical_value"], 4),
        p_value=p2["p_value"],
        ci_lower=round(p2["ci_lower"], 4),
        ci_upper=round(p2["ci_upper"], 4),
        cohens_d=None,
        effect_magnitude="N/A",
        reject_null=dec2["reject_null"],
        decision_pvalue=dec2["decision_pvalue"],
        decision_critical=dec2["decision_critical"],
        decision_ci=dec2["decision_ci"],
        summary="Conversion rate significantly exceeds 10% baseline. Recommendation: rollout variant." if dec2["reject_null"] else "Insufficient evidence of conversion lift."
    )
    results.append(res2)

    # Scenario 3: Manufacturing Quality Control
    # Target weight: 500.0g. Test H0: mu = 500.0 vs H1: mu != 500.0 (Two-tailed)
    df_mfg = loader.load_manufacturing()
    data_mfg = df_mfg["weight_grams"].values
    spec3 = HypothesisSpec(
        test_id="SCENARIO_3_MANUFACTURING_QC",
        scenario_name="Manufacturing High-Precision QC Calibration",
        test_kind=TestKind.ONE_SAMPLE_T,
        parameter_name="mean_product_weight",
        null_value=500.0,
        alternative=AlternativeType.TWO_SIDED,
        alpha=0.05,
        description="Verify whether filling line calibration has drifted from nominal 500.0g target."
    )
    t3 = one_sample_t_test(data_mfg, mu_0=500.0, alpha=0.05, alternative="two-sided")
    d3 = compute_cohens_d(t3["sample_mean"], 500.0, t3["sample_std"])
    dec3 = evaluate_decisions(
        p_value=t3["p_value"],
        test_statistic=t3["test_statistic"],
        critical_value=t3["critical_value"],
        alternative=AlternativeType.TWO_SIDED,
        alpha=0.05,
        ci_lower=t3["ci_lower"],
        ci_upper=t3["ci_upper"],
        null_val=500.0
    )
    res3 = TestResult(
        spec=spec3,
        sample_size=t3["n"],
        point_estimate=round(t3["sample_mean"], 4),
        standard_error=round(t3["standard_error"], 4),
        test_statistic=round(t3["test_statistic"], 4),
        critical_value=round(t3["critical_value"], 4),
        p_value=t3["p_value"],
        ci_lower=round(t3["ci_lower"], 4),
        ci_upper=round(t3["ci_upper"], 4),
        cohens_d=round(d3["cohens_d"], 4),
        effect_magnitude=d3["magnitude"],
        reject_null=dec3["reject_null"],
        decision_pvalue=dec3["decision_pvalue"],
        decision_critical=dec3["decision_critical"],
        decision_ci=dec3["decision_ci"],
        summary="Significant calibration drift detected away from 500.0g. Machine recalibration required." if dec3["reject_null"] else "Filling line is in statistical control."
    )
    results.append(res3)

    # Scenario 4: Customer Order Value Optimization
    # Target basket value: > $120.00. Test H0: mu <= 120.0 vs H1: mu > 120.0 (Right-tailed)
    df_orders = loader.load_customer_orders()
    data_orders = df_orders["order_total_usd"].values
    spec4 = HypothesisSpec(
        test_id="SCENARIO_4_CUSTOMER_ORDER_VALUE",
        scenario_name="Customer Order Basket Value Optimization",
        test_kind=TestKind.ONE_SAMPLE_T,
        parameter_name="mean_order_total_usd",
        null_value=120.0,
        alternative=AlternativeType.GREATER,
        alpha=0.05,
        description="Test whether dynamic bundling lifted average order basket size above $120.00."
    )
    t4 = one_sample_t_test(data_orders, mu_0=120.0, alpha=0.05, alternative="greater")
    d4 = compute_cohens_d(t4["sample_mean"], 120.0, t4["sample_std"])
    dec4 = evaluate_decisions(
        p_value=t4["p_value"],
        test_statistic=t4["test_statistic"],
        critical_value=t4["critical_value"],
        alternative=AlternativeType.GREATER,
        alpha=0.05,
        ci_lower=t4["ci_lower"],
        ci_upper=t4["ci_upper"],
        null_val=120.0
    )
    res4 = TestResult(
        spec=spec4,
        sample_size=t4["n"],
        point_estimate=round(t4["sample_mean"], 4),
        standard_error=round(t4["standard_error"], 4),
        test_statistic=round(t4["test_statistic"], 4),
        critical_value=round(t4["critical_value"], 4),
        p_value=t4["p_value"],
        ci_lower=round(t4["ci_lower"], 4),
        ci_upper=round(t4["ci_upper"], 4),
        cohens_d=round(d4["cohens_d"], 4),
        effect_magnitude=d4["magnitude"],
        reject_null=dec4["reject_null"],
        decision_pvalue=dec4["decision_pvalue"],
        decision_critical=dec4["decision_critical"],
        decision_ci=dec4["decision_ci"],
        summary="Average basket value significantly exceeds $120.00 threshold." if dec4["reject_null"] else "Basket size not significantly greater than $120.00."
    )
    results.append(res4)

    return results
