# Day 55 — Advanced NumPy & Student Analytics Engine V2

Welcome to **Day 55** of the **200 Days of Python Challenge**! Today completes our intensive NumPy module with **Advanced NumPy**, featuring fancy indexing, conditional replacement, multi-condition categorization via `np.select()`, array stacking/splitting, view vs. copy memory management, in-depth broadcasting, NaN-aware aggregations, and basic linear algebra.

---

## 🎯 Day 55 Goals & Topics Covered

- **Fancy / Integer Indexing**: Non-contiguous element extraction across dimensions (`arr[[0, 2, 4]]`, `arr[[r1, r2], [c1, c2]]`).
- **Conditional Replacement & `np.where()`**: Thresholding/imputation (`marks[marks < 40] = 40`) and vectorized ternary branching.
- **Multi-Condition Grading (`np.select()`)**: Sequential condition evaluation for letter grade assignment (`A+`, `A`, `B`, `C`, `D`, `E`, `F`).
- **Sorting & Frequency Counting**: `np.sort()`, `np.argsort()` (ranking indices), `np.unique(..., return_counts=True)`.
- **Array Stacking & Splitting**: `np.vstack()`, `np.hstack()`, `np.concatenate()`, `np.split()`, `np.array_split()`.
- **Copy vs. View**: Memory buffer sharing (`a[1:3]`) vs. deep allocation (`a[1:3].copy()`).
- **Broadcasting**: Multi-row arithmetic broadcasting across 2D/1D dimension bounds.
- **NaN Handling**: `np.nan`, `np.isnan()`, `np.nanmean()`, `np.nansum()`, `np.nanmin()`, `np.nanmax()`, `np.nanargmax()`, `np.nanargmin()`, and missing data imputation.
- **Linear Algebra Basics**: Dot product (`np.dot`), matrix multiplication (`A @ B`, `np.matmul`), and transpose (`.T`).

---

## 📂 Project Architecture

```text
Day 55/
├── Day55.md                        # Masterclass theory notes & 25 interview Q&As
├── README.md                       # Project documentation (this file)
├── pyproject.toml                  # Pytest & tool configurations
├── requirements.txt                # Dependencies (numpy, pytest)
├── app/                            # Student Analytics Engine V2 package
│   ├── __init__.py
│   ├── generator.py                # Reproducible synthetic dataset generator with NaNs
│   ├── validator.py                # Dimension & score range validation (NaN tolerant)
│   ├── analyzer.py                 # Advanced NaN-aware analytics, ranking & np.select grading
│   ├── report.py                   # Executive ASCII report generator
│   └── main.py                     # Pipeline execution & CLI entry point
├── coding_challenges/              # 8 Standalone coding challenges
│   ├── challenge1_fancy_indexing.py
│   ├── challenge2_conditional_replacement.py
│   ├── challenge3_argsort_ranking.py
│   ├── challenge4_unique_counts.py
│   ├── challenge5_nan_handling.py
│   ├── challenge6_broadcasting_discount.py
│   ├── challenge7_matrix_multiplication.py
│   └── challenge8_sales_dataset.py
├── output/                         # Output generated reports
│   └── student_analytics_report.txt# Generated ASCII executive analysis report
└── tests/                          # Automated Pytest suite (21 passing tests)
    ├── conftest.py                 # Test fixtures & sample matrices
    ├── test_generator.py           # Dataset generator & NaN insertion tests
    ├── test_validator.py           # Validation rule & exception tests
    ├── test_analyzer.py            # Analytics, ranking, np.select & NaN tolerance unit tests
    └── test_report.py              # Report generator unit tests
```

---

## 🚀 Student Analytics Engine V2 Features

The Analytics Engine processes a synthetic 100-student x 5-subject marks matrix containing missing `NaN` values:

1. **NaN-Aware Aggregations (`np.nanmean`, `np.nansum`)**: Ignores missing entries cleanly.
2. **Student Rankings (`np.argsort`)**: Descending ranking of students by overall average.
3. **Pass/Fail Branching (`np.where`)**: Element-wise pass ($\ge 40.0\%$) and fail ($< 40.0\%$) metrics.
4. **Letter Grade Classification (`np.select`)**: Multi-condition assignment (`A+` .. `F`).
5. **Grade Distribution (`np.unique`)**: Tallying student counts per grade tier.
6. **Extrema Search (`np.nanargmax`, `np.nanargmin`)**: Highest/lowest performer and best/lowest subject.

---

## 🧪 Testing & Verification

Run the full automated test suite using `pytest`:

```bash
pytest "Day 55/tests/"
```

All 21 test cases verify dataset generation, shape alignment, NaN tolerance, `np.select` grading, `np.where` branching, `np.argsort` ranking, and validation rules.
