# Day 66 — Probability Fundamentals for Data Science

## 🎯 Overview
Day 66 marks the transition from descriptive statistics to probability theory. We explore the mathematical mechanics governing randomness, chance, and predictive modeling: sample spaces, events, union/intersection, conditional probability, Bayes' theorem, base rate fallacies, expected value, discrete vs continuous random variables, and the Law of Large Numbers.

## 📊 Core Concepts Covered
- **Foundations:** Sample spaces, outcomes, events, probability axioms ($0 \le P(A) \le 1$).
- **Set Operations:** Complement ($P(A^c) = 1 - P(A)$), Union ($A \cup B$), Intersection ($A \cap B$), Addition Rule.
- **Event Relationships:** Mutually exclusive vs Independent vs Dependent events.
- **Sampling:** With replacement vs Without replacement.
- **Conditional Probability:** $P(A|B) = \frac{P(A \cap B)}{P(B)}$.
- **Bayes' Theorem & Base Rate Fallacy:** $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$.
- **Expected Value:** $E(X) = \sum x_i P(x_i)$ for discrete random variables.
- **Convergence & Simulation:** Law of Large Numbers, empirical frequencies, and Monte Carlo estimation.

## 📁 Directory Structure
```text
Day 66/
├── Day66.md                   # Masterclass Notes, 25 Interview Q&As & Assessment Solutions
├── app/                       # Main Project: Real-World Probability Simulator
│   ├── main.py                # Pipeline CLI Entry Point
│   ├── analysis.py            # Simulation Aggregation & Comparison
│   ├── visualizations.py      # 5 Diagnostic Visualizations (300 DPI)
│   ├── report.py              # ASCII Summary Report & CSV Exporter
│   ├── probability/           # Theoretical Calculation Modules
│   │   ├── basic.py
│   │   ├── conditional.py
│   │   ├── bayes.py
│   │   └── expected_value.py
│   └── simulations/           # Empirical Simulation Engines
│       ├── coin.py
│       ├── dice.py
│       ├── cards.py
│       ├── conversion.py
│       └── risk.py
├── exercises/                 # 9 Practical Hands-on Tasks
├── coding_challenges/         # 5 Advanced Probability Challenges
├── data/                      # E-Commerce Sales Dataset (750 rows)
├── output/                    # Generated Reports & Visualizations
│   ├── charts/                # Publication-Grade Diagnostic Figures
│   ├── coin_results.csv
│   ├── dice_results.csv
│   ├── card_results.csv
│   ├── conversion_results.csv
│   └── probability_report.txt
├── tests/                     # 30+ Pytest Automation Test Cases
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 🚀 How to Run
```bash
# Run tests
pytest

# Run the Probability Simulator Engine
python -m app.main
```
