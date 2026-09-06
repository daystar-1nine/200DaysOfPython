# Day 67 — Probability Distributions

## Topics Covered
- Probability Distributions (Discrete vs Continuous)
- PMF (Probability Mass Function)
- PDF (Probability Density Function)
- CDF (Cumulative Distribution Function)
- PPF (Percent Point Function / Quantiles)
- Bernoulli Distribution
- Binomial Distribution
- Uniform Distribution (Continuous)
- Normal Distribution (Gaussian & Standard Normal)
- Poisson Distribution
- Expected Value & Variance
- Distribution Parameters
- Random Sampling with SciPy & NumPy
- Theoretical vs Simulated Comparison (Law of Large Numbers)

## Main Project: Probability Distribution Analyzer
A modular, high-performance distribution analysis package providing:
- Parameter validation & mathematical properties
- Probability queries (`pmf_or_pdf`, `cdf`, `ppf`)
- Stochastic simulation & empirical distribution sampling
- Theoretical vs experimental comparison metrics
- 8 publication-grade visualization figures
- Exported CSV datasets & executive ASCII report

## Project Structure
```text
Day 67/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── validator.py
│   ├── analyzer.py
│   ├── simulator.py
│   ├── visualizations.py
│   ├── report.py
│   └── main.py
├── distributions/
│   ├── __init__.py
│   ├── base.py
│   ├── bernoulli.py
│   ├── binomial.py
│   ├── uniform.py
│   ├── normal.py
│   └── poisson.py
├── exercises/
├── coding_challenges/
├── output/
│   ├── charts/
│   ├── distribution_summary.csv
│   ├── probability_results.csv
│   ├── simulation_results.csv
│   └── distribution_report.txt
└── tests/
```

## Testing
Comprehensive pytest suite covering all distribution classes, simulator, validator, and analytics pipeline.
```bash
pytest tests -v
```

## Progress
- **Day 67 / 200 Days Completed (33.5%)**
