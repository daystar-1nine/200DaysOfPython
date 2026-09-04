# Day 54 — NumPy Fundamentals & Student Performance Analyzer

Welcome to **Day 54** of the **200 Days of Python Challenge**! Today marks our entry into the Data Science ecosystem with **NumPy Fundamentals**, introducing high-performance N-dimensional array (`ndarray`) computing, vectorized operations, broadcasting, reshaping, and 2D axis aggregations.

---

## 🎯 Day 54 Goals & Topics Covered

- **NumPy & `ndarray`**: Understanding contiguous C memory buffers vs. standard Python lists, homogeneous `dtype`, `ndim`, `shape`, `size`.
- **Indexing & Slicing**: 1D and 2D sub-matrix slicing (`[row, col]`, `[:, :2]`).
- **Vectorized Computation**: High-performance scalar and element-wise arithmetic (`+`, `-`, `*`, `/`, `**`) without Python `for` loops.
- **Broadcasting**: Array shape alignment rules for implicit expansion.
- **Boolean Indexing & Filtering**: Mask creation using bitwise operators (`&`, `|`, `~`) and conditional filtering.
- **Array Generation Helpers**: `np.arange()`, `np.linspace()`, `np.zeros()`, `np.ones()`, `np.full()`, `np.eye()`.
- **Reshaping & Transpose**: `reshape(rows, cols)`, `reshape(-1)`, and `.T` matrix transposition.
- **2D Axis Aggregations**: `np.sum()`, `np.mean()`, `np.std()`, `np.median()`, `np.min()`, `np.max()`, `np.argmax()`, `np.argmin()`, `np.argsort()` across `axis=0` (column-wise) and `axis=1` (row-wise).
- **Min-Max Feature Normalization**: Implementing $\frac{x - x_{\min}}{x_{\max} - x_{\min}}$ without external libraries.
- **Random Number Generation**: Reproducible random state using `np.random.default_rng(seed)`.

---

## 📂 Project Architecture

```text
Day 54/
├── Day54.md                        # Masterclass theory notes & 25 interview Q&As
├── README.md                       # Project documentation (this file)
├── pyproject.toml                  # Pytest & tool configurations
├── requirements.txt                # Dependencies (numpy, pytest)
├── app/                            # Student Performance Analyzer package
│   ├── __init__.py
│   ├── data.py                     # Seed dataset & validation rules
│   ├── analyzer.py                 # Vector analytics, averages, rankings & normalization
│   ├── report.py                   # ASCII performance report generator
│   └── main.py                     # Interactive CLI entry point & driver
├── coding_challenges/              # 6 Standalone coding challenges
│   ├── challenge1_array_basics.py  # 1D array creation & statistics (1..100)
│   ├── challenge2_even_numbers.py  # Boolean mask filtering without for-loops
│   ├── challenge3_matrix.py        # 3x3 matrix transposition & axis sums
│   ├── challenge4_temperature.py   # Temperature time-series analytics
│   ├── challenge5_student_marks.py # 2D student marks matrix analysis
│   └── challenge6_normalization.py # Min-Max score normalization formula
├── exercises/                      # Performance benchmarking exercises
│   └── exercise1_vectorization_benchmark.py # Python loop vs NumPy 1M vector speedup
├── output/                         # Output generated reports
│   └── performance_report.txt      # Generated ASCII performance report
└── tests/                          # Automated Pytest suite (16 passing tests)
    ├── conftest.py                 # Test fixtures & sample matrices
    ├── test_data.py                # Dataset shape & boundary validation tests
    └── test_analyzer.py            # Vector analytics, rankings & normalization tests
```

---

## 🚀 Performance Analyzer Features

The Student Performance Analyzer computes numerical metrics on a 2D student marks matrix:

1. **Student Averages & Totals (`axis=1`)**: Row-wise aggregate performance metrics.
2. **Subject Performance Averages (`axis=0`)**: Column-wise subject difficulty metrics.
3. **Student Ranking (`np.argsort`)**: Descending order ranking of student performance.
4. **Top & Lowest Performer (`np.argmax`, `np.argmin`)**: Extremal student identification.
5. **Best Subject (`np.argmax`)**: Subject with highest overall class mean score.
6. **Boolean Filtering**: High performers ($\ge 80\%$) and low performers ($< 60\%$).
7. **Min-Max Score Normalization**: Matrix scaling into uniform $[0.0, 1.0]$ range.

---

## 🧪 Testing & Verification

Run the full automated test suite using `pytest`:

```bash
pytest "Day 54/tests/"
```

All 16 test cases verify array shapes, dimensions, axis aggregations, boolean filtering, ranking, normalization, and validation rules.
