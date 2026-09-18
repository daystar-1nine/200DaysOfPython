<div align="center">

# 🚀 Day 100 / 200 — 50% MILESTONE

**End-to-End AI Data Analytics Pipeline Capstone**

[![Milestone](https://img.shields.io/badge/Status-50%25%20Complete-green.svg)](#)
[![Pipeline](https://img.shields.io/badge/Architecture-E2E%20Pipeline-blueviolet.svg)](#)

</div>

---

## 📌 Overview

Today marks **Day 100** of the 200-day Python + Data Science challenge. 

Rather than learning a new specific sub-library, this day serves as a **Mini-Capstone**. The goal was to prove integration capabilities: bridging the gap between raw CSV data loading, Pandas cleaning, statistical feature engineering, and Scikit-Learn Model evaluation within a professionally structured Python repository.

---

## 🎯 Capstone Objectives

1. **Integrated Pipeline**: Linking `data` -> `cleaning` -> `features` -> `model` -> `evaluation`.
2. **Robust Testing**: Utilizing Pytest to explicitly assert `NaN` imputation logic and outlier eviction.
3. **Automated Visualization**: Generating 12 standardized milestone plots (ROC, Heatmaps, Feature Importance).
4. **Professional Architecture**: Strict separation of concerns (Data loading is isolated from Model training).

---

## 🏗️ Technical Architecture

```text
  Raw CSV Data
       │
  Data Loader & Cleaner (Pandas)
       │
  EDA Statistics
       │
  Feature Engineering (StandardScaler / Train-Test Split)
       │
  RandomForestClassifier (Scikit-Learn)
       │
  Evaluation Metrics (Precision/Recall/F1)
       │
  Visualization Generator (Matplotlib)
```

---

## 🚀 How to Run

1. **Enter Directory**:
   ```bash
   cd "Day 100"
   ```
2. **Execute Full Analytics Pipeline**:
   ```bash
   python -m app.main
   ```
3. **Execute Testing Assertions**:
   ```bash
   python -m pytest tests/
   ```
