# Day 71 — Two-Sample Tests & A/B Testing Analysis Engine

## Overview
A production-grade experimentation and two-sample statistical inference engine for analyzing A/B tests and comparative experiments:
- Independent Two-Sample Tests (Student's t-test and Welch's unequal-variance t-test)
- Paired Two-Sample t-Tests for within-subject repeated measures
- Two-Proportion Pooled Z-Tests for conversion rate comparison
- Confidence Intervals for Differences (Mean Differences & Proportion Differences)
- Effect Size Quantification (Cohen's d for independent/paired, Absolute & Relative Lift)
- Financial Impact Modeling (Incremental conversions, projected annual revenue)
- Comprehensive Guardrail Metric Audits (Bounce rate, Session duration, Refund rate)
- Robust Decision Engine evaluating statistical significance, practical lift, and guardrail constraints
- 8 Publication-Grade Diagnostic Visualizations
- 40+ Automated Pytest Unit and Integration Tests

## Project Structure
```text
Day 71/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── generator.py
│   ├── validator.py
│   ├── metrics.py
│   ├── mean_tests.py
│   ├── proportion_tests.py
│   ├── confidence_intervals.py
│   ├── effect_size.py
│   ├── business_impact.py
│   ├── decision.py
│   ├── visualizations.py
│   ├── report.py
│   └── main.py
├── practice/
│   ├── task1_independent_ttest.py
│   ├── task2_paired_ttest.py
│   ├── task3_compare_test_types.py
│   ├── task4_difference_ci.py
│   ├── task5_ab_conversion.py
│   ├── task6_sample_size_impact.py
│   └── task7_simulate_ab_test.py
├── coding_challenges/
│   ├── challenge1_compare_means.py
│   ├── challenge2_compare_proportions.py
│   ├── challenge3_ab_test_report.py
│   └── challenge4_simulate_false_positives.py
├── data/
│   ├── experiment_config.json
│   └── experiment_users.csv
├── output/
│   ├── charts/
│   ├── experiment_summary.csv
│   ├── statistical_results.csv
│   ├── business_impact.txt
│   └── ab_test_report.txt
└── tests/
```

## Running the Application
```bash
python -m app.main
```

## Running Tests
```bash
pytest tests -v
```

## Progress
- **Day 71 / 200 Days Completed (35.5%)**
- **Milestone:** Advancing into Experimental Design & A/B Testing!
