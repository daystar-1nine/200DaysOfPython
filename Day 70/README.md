# Day 70 — Hypothesis Testing Fundamentals Engine

## Overview
A production-grade statistical hypothesis testing and inference engine implementing:
- Parametric Tests: 1-Sample Z-Test (known $\\sigma$), 1-Sample Student's t-Test (unknown $\\sigma$)
- Proportion Tests: 1-Sample Z-Test for Proportions (Wald & Wilson Continuity)
- Dual Decision Logic: Critical Value Rejection Regions vs P-Value thresholds
- Confidence Interval Duality: Linkage between $(1 - \\alpha)$ CIs and two-sided tests
- Practical Significance: Cohen's d Effect Size benchmarks ($d = \\frac{\\bar{x} - \\mu_0}{s}$)
- Type I Error Monte Carlo Simulation
- 4 Real-World Business Scenarios: Delivery Times, Website Conversion, Manufacturing QC, Customer Order Values
- 7 Publication-Grade Visualizations
- 40+ Automated Pytest Unit and Integration Tests

## Project Structure
```text
Day 70/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── validator.py
│   ├── loader.py
│   ├── stats_calc.py
│   ├── hypotheses.py
│   ├── tests_mean.py
│   ├── tests_proportion.py
│   ├── confidence_intervals.py
│   ├── effect_size.py
│   ├── decision.py
│   ├── scenarios.py
│   ├── visualizations.py
│   ├── report.py
│   └── main.py
├── practice/
│   ├── task1_manual_hypothesis_test.py
│   ├── task2_one_sample_z_test.py
│   ├── task3_one_sample_t_test.py
│   ├── task4_one_tailed_vs_two_tailed.py
│   ├── task5_proportion_test.py
│   ├── task6_cohen_d.py
│   └── task7_type1_error_simulation.py
├── coding_challenges/
│   ├── challenge1_one_sample_ttest.py
│   ├── challenge2_one_sample_ztest.py
│   ├── challenge3_proportion_test.py
│   ├── challenge4_plot_hypothesis_test.py
│   └── challenge5_type1_error_simulation.py
├── data/
│   ├── delivery_times.csv
│   ├── conversion_data.csv
│   ├── manufacturing.csv
│   └── customer_orders.csv
├── output/
│   ├── charts/
│   ├── hypothesis_test_results.csv
│   └── statistical_report.txt
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
- **Day 70 / 200 Days Completed (35.0%)**
- **Milestone:** Over one-third of the journey completed!
