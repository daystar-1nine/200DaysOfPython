# Day 72 — Business Relationship & Correlation Analyzer

## 📊 Overview
A production-grade statistical and diagnostic analysis tool for discovering, evaluating, and visualizing relationships between quantitative variables. The application combines parametric (Pearson $r$) and non-parametric (Spearman $\rho$) metrics, Fisher $z$-transform confidence intervals, sample covariance analysis, multicollinearity screening, outlier leverage diagnostics, non-causal insight synthesis, and 9 publication-grade visualizations.

---

## 🎯 Core Concepts Mastered
- **Covariance:** Joint variation, sample vs. population formulation, Bessel's correction, and unit-scale limitations.
- **Pearson Correlation ($r$):** Standardized covariance ($-1 \le r \le +1$), geometric cosine projection, $t$-statistic transformation ($df = n - 2$), and Fisher's $z$-transformation for confidence intervals.
- **Spearman Rank Correlation ($\rho$):** Non-parametric monotonic association, rank assignment algorithm with average tie handling, and resistance to extreme leverage points.
- **Anscombe's Quartet:** Proof that identical summary statistics ($r = 0.816$, regression fits) can mask wildly divergent distributions.
- **Simpson's Paradox:** Mathematical conditions under which subgroup trends invert upon aggregation.
- **Confounding & Partial Correlation:** Demonstrating how common causes (e.g. Temperature) generate spurious zero-order associations that collapse when controlled for ($\rho_{XY \cdot Z} \approx 0$).
- **Multicollinearity:** Singularity of $(X^T X)^{-1}$, Variance Inflation Factor (VIF), and screening thresholds ($|r| \ge 0.75$).
- **Causal Governance:** Strict avoidance of causal claims based on observational correlation.

---

## 🏗️ Architecture & Modules

```text
Day 72/
├── Day72.md                              # Theoretical Masterclass & 30 Technical Interview Q&As
├── practice/
│   ├── task1_manual_covariance.py        # First-principles covariance calculation vs np.cov
│   ├── task2_pearson_correlation.py      # Pearson r manual formula vs SciPy pearsonr
│   ├── task3_spearman_correlation.py     # Spearman rank correlation vs SciPy spearmanr
│   ├── task4_positive_correlation.py     # Strong positive linear simulation & Fisher CI
│   ├── task5_negative_correlation.py     # Strong negative linear simulation & p-value
│   ├── task6_nonlinear_relationship.py   # Parabolic curve (y = x^2) & Pearson failure
│   ├── task7_outlier_experiment.py       # Single-outlier leverage & Spearman resistance
│   └── task8_correlation_matrix.py       # Pairwise matrix computation on e-commerce dataset
├── coding_challenges/
│   ├── challenge1_correlation_vs_causation.py # Confounder simulation & partial correlation
│   ├── challenge2_simpsons_paradox.py         # Subgroup vs aggregate conversion inversion
│   ├── challenge3_anscombes_quartet.py        # 4 classic datasets & 4-panel figure
│   └── challenge4_robust_correlation.py       # Pearson vs Spearman vs Kendall vs Winsorized
├── data/
│   ├── ecommerce_sales.csv               # 1,200 transactions across 12 numerical features
│   └── ecommerce_sales.json              # Equivalent JSON dataset for multi-format testing
├── app/
│   ├── __init__.py                       # Package initialization
│   ├── config.py                         # AppConfig, thresholds, and strength enums
│   ├── loader.py                         # CSV and JSON format loader
│   ├── cleaner.py                        # Numerical filtering, inf and null handling
│   ├── validator.py                      # Dimension and constant column validator
│   ├── covariance.py                     # Sample/population covariance matrix engine
│   ├── pearson.py                        # Pearson r, t-test, and Fisher z 95% CI
│   ├── spearman.py                       # Spearman rho and rank transformation engine
│   ├── correlation_matrix.py             # Full pairwise matrix & difference calculations
│   ├── outliers.py                       # Univariate IQR and bivariate leverage analysis
│   ├── analyzer.py                       # High-level synthesis & 10 business Q&A engine
│   ├── insights.py                       # Non-causal plain-English insights generator
│   ├── visualizations.py                 # 9 publication-grade matplotlib charts (Agg)
│   ├── report.py                         # ASCII report and CSV export engine
│   └── main.py                           # CLI pipeline orchestrator
├── output/
│   ├── covariance_matrix.csv             # Unbiased sample covariance matrix
│   ├── correlation_matrix.csv            # Pairwise Pearson correlation matrix
│   ├── correlation_results.csv           # Detailed table of all 66 variable pairs
│   ├── relationship_report.txt           # Formatted ASCII executive report
│   └── charts/
│       ├── correlation_heatmap.png       # Upper-triangle masked correlation heatmap
│       ├── revenue_profit.png            # Revenue vs Profit regression fit
│       ├── quantity_revenue.png          # Quantity vs Revenue scatter
│       ├── discount_profit.png           # Discount vs Profit inverse association
│       ├── cost_revenue.png              # Cost vs Revenue co-movement
│       ├── pearson_spearman.png          # Pearson vs Spearman parity plot
│       ├── strongest_positive.png        # Top positive correlations bar chart
│       ├── strongest_negative.png        # Top negative correlations bar chart
│       └── dashboard.png                 # Executive 4-panel correlation dashboard
├── tests/                                # 40+ unit and integration tests
├── requirements.txt                      # Project dependencies
├── pyproject.toml                        # Pytest configuration
├── pytest.ini                            # Pytest path settings
└── README.md                             # Documentation & user guide
```

---

## 🚀 How to Run the Code

### 1. Execute Full Application Pipeline
```bash
python "Day 72/app/main.py"
```

### 2. Run All Practice Tasks
```bash
python "Day 72/practice/task1_manual_covariance.py"
python "Day 72/practice/task2_pearson_correlation.py"
python "Day 72/practice/task3_spearman_correlation.py"
python "Day 72/practice/task4_positive_correlation.py"
python "Day 72/practice/task5_negative_correlation.py"
python "Day 72/practice/task6_nonlinear_relationship.py"
python "Day 72/practice/task7_outlier_experiment.py"
python "Day 72/practice/task8_correlation_matrix.py"
```

### 3. Run Advanced Coding Challenges
```bash
python "Day 72/coding_challenges/challenge1_correlation_vs_causation.py"
python "Day 72/coding_challenges/challenge2_simpsons_paradox.py"
python "Day 72/coding_challenges/challenge3_anscombes_quartet.py"
python "Day 72/coding_challenges/challenge4_robust_correlation.py"
```

### 4. Run Pytest Test Suite
```bash
pytest "Day 72/tests" -v
```

---

## 📈 Key Findings from E-Commerce Analysis
1. **Core Positive Driver:** `Quantity` strongly drives `Revenue` ($r = 0.654, p < 0.001$).
2. **Margin Erosion Warning:** `Discount` exhibits a negative correlation with `Profit` ($r = -0.266$) and `Profit_Margin` ($r = -0.477$).
3. **Multicollinearity Alert:** `Unit_Price` and `Cost_Price` have an extreme correlation of $r = 0.996$, presenting severe redundancy in predictive modeling.
4. **Non-Linear Divergence:** `Marketing_Spend` vs `Revenue` yields $r = 0.233$ but $\rho = 0.968$ (difference = $0.734$), revealing a non-linear diminishing returns curve.
5. **Causal Governance:** All findings represent statistical associations and do not prove causation without randomized A/B experimentation.
