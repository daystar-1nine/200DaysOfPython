# Day 68 — Inferential Statistics: Sampling, Sampling Distributions & Central Limit Theorem

## Overview
A comprehensive Python application demonstrating the principles of inferential statistics:
- Population vs Sample ($\mu, \sigma$ vs $\bar{x}, s$)
- Parameter vs Statistic
- Sampling methodologies (Simple Random, Systematic, Stratified, Cluster)
- Sampling Bias and its consequences
- Sampling Distribution of the Sample Mean
- Standard Error ($SE = \sigma / \sqrt{n}$)
- Central Limit Theorem (CLT)
- Law of Large Numbers (LLN) vs Central Limit Theorem (CLT)
- Confidence Interval Intuition
- Non-Parametric Bootstrap Resampling

## Main Project: Sampling & Central Limit Theorem Simulator
A production-grade Python package featuring:
- Configurable population generation across 5 distributions (Normal, Uniform, Exponential, Binomial, Poisson)
- Decoupled sampling engines with/without replacement and stratification
- Standard Error scaling analysis across multi-scale sample batches
- CLT convergence analysis across sample sizes $n \in [5, 10, 30, 50, 100]$
- Non-parametric Bootstrap resampling engine with empirical percentile intervals
- 7 publication-grade diagnostic figures
- Exported CSV datasets and executive ASCII statistical report

## Project Structure
```text
Day 68/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── population.py
│   ├── sampler.py
│   ├── stats_engine.py
│   ├── sampling_distribution.py
│   ├── clt.py
│   ├── bootstrap.py
│   ├── analyzer.py
│   ├── visualizations.py
│   ├── report.py
│   └── main.py
├── practice/
├── coding_challenges/
├── data/
│   └── sample_data.csv
├── output/
│   ├── charts/
│   ├── sampling_results.csv
│   ├── bootstrap_results.csv
│   └── statistics_report.txt
└── tests/
```

## Testing
```bash
pytest tests -v
```

## Progress
- **Day 68 / 200 Days Completed (34.0%)**
