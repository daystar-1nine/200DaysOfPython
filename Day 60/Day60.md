# 🚀 DAY 60 — SECOND MAJOR MILESTONE
🏆 Python + Backend + NumPy + Pandas + EDA Assessment & Capstone

Congratulations on completing **60 Days** of the 200 Days of Python Challenge! 🎉
This milestone marks the completion of your comprehensive **Data Science Foundation** phase (Days 51–60).

---

## 🧭 Milestone Architecture & Road Map

```text
                                  DAY 60 MILESTONE
                                         │
        ┌───────────────────┬────────────┴────────────┬───────────────────┐
        ↓                   ↓                         ↓                   ↓
   REVISION            THEORY TEST               CODING TESTS      CAPSTONE PROJECT
  (Days 51-59)         (40 Questions)          (Python/NumPy/      (BI Analytics Engine
                                                Pandas/Cleaning/    + 30+ Pytest Tests)
                                                EDA Practicals)
```

---

## 🧠 Part 1: Comprehensive Revision (Days 51–59)

### 1. Python Data Processing (Days 51–53)
- **Type Hints**: Function and variable annotations (`list[float]`, `dict[str, int]`, `Optional[T]`, `Union[A, B]`) ensure code readability, IDE autocomplete support, and static analysis verification via MyPy.
- **Dataclasses**: Decorator `@dataclass` generates boilerplate methods (`__init__`, `__repr__`, `__eq__`). Using `field(default_factory=list)` prevents shared mutable state across instances.
- **Enums**: `Enum` classes enforce fixed sets of categorical values, eliminating hardcoded string typos.
- **Pathlib**: `Path` objects provide cross-platform path manipulation, preventing OS-dependent string concatenation issues.
- **Collections**: `Counter` tallies elements; `defaultdict` provides missing key defaults without `KeyError`; `deque` supports $O(1)$ appends and pops from both ends.
- **Serialization (JSON & CSV)**:
  - `json.dumps()` / `json.loads()` for in-memory string serialization/deserialization.
  - `json.dump()` / `json.load()` for file-based serialization/deserialization.
  - `csv.DictReader` / `csv.DictWriter` read/write tabular files as Python dictionaries. Since CSV inputs are text strings, numbers and dates must be cast explicitly.
- **Standard Pipeline**: $\text{READ} \to \text{CLEAN} \to \text{VALIDATE} \to \text{TRANSFORM} \to \text{ANALYZE} \to \text{REPORT}$.

### 2. NumPy Foundations & Advanced Operations (Days 54–55)
- **ndarray**: High-performance contiguous memory array with homogeneous data types (`dtype`), dimensionality (`ndim`), element count (`size`), and shape tuple (`shape`).
- **Vectorization & Broadcasting**: Operations occur in compiled C loops without Python overhead. Broadcasting extends lower-dimensional arrays across higher-dimensional arrays following trailing-dimension compatibility rules.
- **Axis Mechanics**:
  - `axis=0`: Collapses across rows (computes column-wise statistics).
  - `axis=1`: Collapses across columns (computes row-wise statistics).
- **Views vs Copies**: Slicing an array creates a memory view (mutating the slice mutates the source array); fancy indexing or `.copy()` creates an independent memory copy.
- **Conditional Logic**: `np.where(condition, x, y)` and `np.select(condlist, choicelist)`.
- **NaN Handling**: IEEE 754 floating-point NaNs; handled via `np.isnan()`, `np.nanmean()`, `np.nansum()`.
- **Linear Algebra**: Matrix dot product via `A @ B` or `np.matmul()`.

### 3. Pandas Fundamentals, Manipulation & Cleaning (Days 56–58)
- **Series & DataFrame**: 1D labeled array and 2D tabular labeled data structure.
- **Indexing**: `loc` uses label-based indexing (inclusive of end bounds); `iloc` uses integer-position-based indexing (exclusive of end bounds).
- **Aggregations & Grouping**: `groupby()` splits data into groups, applies aggregation functions (`sum`, `mean`, `count`, `min`, `max`), and combines results.
- **Combining Datasets**: `merge()` performs relational joins (inner, left, right, outer); `concat()` stacks DataFrames vertically or horizontally.
- **Cleaning Mechanics**:
  - Nulls: `isna()`, `dropna()`, `fillna()`, `ffill()`, `bfill()`.
  - Duplicates: `duplicated(subset, keep)`, `drop_duplicates()`.
  - Type parsing: `pd.to_numeric(errors="coerce")`, `pd.to_datetime(format="mixed", errors="coerce")`.

### 4. Advanced Pandas & EDA Foundations (Day 59)
- **`groupby().transform()`**: Returns row-aligned values of identical length to the original DataFrame, enabling broadcast comparisons against group averages.
- **Intra-group `rank()`**: Ranks items within categorical partitions (`method="dense"` preserves consecutive integers).
- **Time-Series Shifts**: `shift(1)` accesses previous row values; `diff()` computes lag differences; `pct_change()` computes percentage growth.
- **Rolling Windows**: `rolling(window=N)` smoothes fluctuations using moving averages.
- **Outlier Detection**: Interquartile Range ($IQR = Q3 - Q1$). Lower Bound: $Q1 - 1.5 \times IQR$; Upper Bound: $Q3 + 1.5 \times IQR$.
- **Bivariate Correlation**: `df.corr(numeric_only=True)` computes Pearson linear relationship coefficients between $[-1, 1]$.

---

## 📝 Part 5: Complete 40 Theory Assessment Questions & Answers

### 🔵 Section A — Python (Q1–Q10)

1. **What problem do type hints solve?**
   - Type hints provide explicit documentation of function parameter and return types, improve developer experience with IDE autocomplete and refactoring, and allow static type checkers (such as MyPy) to catch type-mismatch bugs prior to runtime execution.

2. **Why use a dataclass instead of manually writing a simple class?**
   - A `@dataclass` automatically generates standard boilerplate methods—including `__init__()`, `__repr__()`, `__eq__()`, and optionally ordering methods—reducing human error and code verbosity while keeping classes maintainable and pythonic.

3. **What is the purpose of `default_factory`?**
   - `default_factory` provides a callable (such as `list`, `dict`, or a custom factory function) to generate fresh default values for mutable fields. This prevents the common Python trap where all dataclass instances share the exact same mutable object in memory.

4. **What is an Enum?**
   - An `Enum` (Enumeration) is a symbolic name bound to unique, constant values. Enums prevent hardcoded strings (magic strings), enforce valid domain values, and prevent typos in categorical data.

5. **Why is `pathlib` useful?**
   - `pathlib` provides an object-oriented API for filesystem paths. It handles OS-specific path separators automatically (e.g. `\` on Windows vs `/` on POSIX), enables intuitive path concatenation via the `/` operator, and encapsulates common filesystem operations (such as `.exists()`, `.mkdir()`, `.read_text()`) directly on path objects.

6. **What is the difference between `Counter` and `defaultdict`?**
   - `Counter` is specifically designed for tallying hashable objects; accessing a missing key returns `0` without adding that key to the dictionary. `defaultdict` initializes any specified default type (such as `list`, `int`, `set`) whenever a non-existent key is accessed and immediately stores that key in the dictionary.

7. **What is JSON serialization?**
   - JSON serialization (also called encoding or marshaling) is the process of translating Python in-memory data structures (such as dictionaries, lists, strings, and numbers) into a standardized UTF-8 text string formatted according to the JSON specification.

8. **What is the difference between `json.dump()` and `json.dumps()`?**
   - `json.dump()` serializes a Python object directly to an open writable file stream (`.json` file). `json.dumps()` ("dump string") serializes a Python object into an in-memory Python `str`.

9. **What is the difference between `json.load()` and `json.loads()`?**
   - `json.load()` reads and deserializes JSON data from an open readable file stream. `json.loads()` ("load string") parses and deserializes JSON data from an in-memory Python `str` or bytes object.

10. **Why do CSV values often need type conversion?**
    - The CSV specification has no native type system; all fields are stored and ingested as raw text strings. Numbers, booleans, and dates must be explicitly parsed into `int`, `float`, `bool`, or `datetime` objects before performing calculations.

---

### 🟢 Section B — NumPy (Q11–Q20)

11. **What is an `ndarray`?**
    - An `ndarray` (N-dimensional array) is NumPy's core data structure: a homogeneous multidimensional container of fixed-size elements stored in contiguous blocks of memory, designed for fast vectorized numerical operations.

12. **What does `shape` tell you?**
    - The `shape` attribute returns a tuple of integers indicating the size of the array along each dimension (e.g., `(rows, columns)` for a 2D array).

13. **What is the difference between `size` and `shape`?**
    - `shape` is a tuple representing dimensions and lengths along each axis (e.g., `(3, 4)`). `size` is a single integer representing the total number of elements across all dimensions ($3 \times 4 = 12$).

14. **What is vectorization?**
    - Vectorization is the delegation of element-wise array operations to highly optimized, pre-compiled C and Fortran loops, eliminating the performance overhead, dynamic typing, and pointer chasing of standard Python `for` loops.

15. **What is broadcasting?**
    - Broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations. The smaller array is virtually stretched or repeated across the larger array without copying data in memory, provided their trailing dimensions either match or one of them equals 1.

16. **What does `axis=0` mean for a 2D array?**
    - `axis=0` operates along the rows (downwards vertically). Aggregating with `axis=0` collapses the rows to compute column-wise statistics.

17. **What does `axis=1` mean for a 2D array?**
    - `axis=1` operates along the columns (horizontally across). Aggregating with `axis=1` collapses the columns to compute row-wise statistics.

18. **What is the difference between a view and a copy?**
    - A **view** is a new array object sharing the exact same underlying memory buffer as the original array; modifying elements in a view mutates the original data. A **copy** allocates separate memory; changes in a copy do not affect the original array.

19. **What does `np.where()` do?**
    - `np.where(condition, [x, y])` performs vectorized ternary selection: it returns elements chosen from `x` where `condition` is True, and from `y` where `condition` is False. If called with only `condition`, it returns indices of truthy elements.

20. **What is the difference between `@` and `*` for matrices?**
    - `*` performs element-wise multiplication between compatible array shapes (Hadamard product). `@` (or `np.matmul()`) performs formal linear algebra matrix multiplication (dot product of rows and columns).

---

### 🟡 Section C — Pandas (Q21–Q30)

21. **What is a `Series`?**
    - A `Series` is a one-dimensional labeled array capable of holding data of any homogeneous type, backed by an explicit index.

22. **What is a `DataFrame`?**
    - A `DataFrame` is a two-dimensional, size-mutable tabular data structure with labeled axes (rows and columns), composed of an aligned collection of Series sharing a common index.

23. **What is the difference between `loc` and `iloc`?**
    - `loc` is label-based indexing (referencing row/column index names, inclusive of endpoint boundaries). `iloc` is integer-position-based indexing (referencing 0-based integer positions, exclusive of endpoint boundaries).

24. **What does `groupby()` do?**
    - `groupby()` implements the Split-Apply-Combine pattern: it splits a DataFrame into distinct subsets based on one or more grouping keys, applies an operation or aggregation function, and combines results into a new data structure.

25. **What is the difference between `groupby().mean()` and `groupby().transform("mean")`?**
    - `groupby().mean()` aggregates and collapses the rows, returning a reduced Series indexed by unique group labels. `groupby().transform("mean")` calculates group means and broadcasts them back across the original row index, returning a Series of identical length to the original DataFrame.

26. **What is the difference between `merge()` and `concat()`?**
    - `merge()` combines DataFrames horizontally based on shared key columns (equivalent to SQL relational joins). `concat()` stacks DataFrames vertically or horizontally along an axis based on index alignment without evaluating relational key values.

27. **What is a pivot table?**
    - A pivot table is a multidimensional summary table that reshapes tabular data by grouping across specified row and column dimensions and aggregating continuous numerical metrics at the intersections.

28. **What does `apply()` do?**
    - `apply()` invokes a custom Python function along an axis of a DataFrame (`axis=0` for columns, `axis=1` for rows) or across each element in a Series.

29. **When would you use `map()`?**
    - `map()` is used on a `Series` to substitute each value with another value using a dictionary mapping, a Series mapping, or an element-wise mapping function.

30. **How do you detect missing values in Pandas?**
    - Use `df.isna()` or `df.isnull()` to generate a boolean mask of null locations, and chain `.sum()` (`df.isna().sum()`) to count missing values per column.

---

### 🔴 Section D — EDA (Q31–Q40)

31. **What is Exploratory Data Analysis (EDA)?**
    - EDA is an iterative approach to analyzing datasets using summary statistics, distribution metrics, and visualizations to understand underlying structures, identify patterns, spot anomalies, and test hypotheses.

32. **What is an outlier?**
    - An outlier is an observation point that lies an abnormal distance from other values in a random sample from a population, significantly deviating from the overall distribution.

33. **What is IQR?**
    - IQR (Interquartile Range) is a measure of statistical dispersion equal to the difference between the 75th percentile ($Q3$) and the 25th percentile ($Q1$): $IQR = Q3 - Q1$.

34. **How do you calculate IQR and outlier boundaries?**
    - Calculate $Q1$ (25th percentile) and $Q3$ (75th percentile).
    - $IQR = Q3 - Q1$.
    - $\text{Lower Bound} = Q1 - 1.5 \times IQR$.
    - $\text{Upper Bound} = Q3 + 1.5 \times IQR$.

35. **Does every outlier need to be removed?**
    - No. Outliers frequently represent genuine business events (such as corporate bulk purchases, viral traffic, or fraud). Automatically deleting valid outliers distorts real-world variance and biases statistical models.

36. **What does correlation measure?**
    - Correlation measures the strength and direction of a linear relationship between two continuous variables on a scale from $-1.0$ (perfect negative) to $+1.0$ (perfect positive).

37. **Does correlation imply causation?**
    - No. Two variables may be strongly correlated due to coincidence or a third confounding variable (spurious correlation) without one causing the other.

38. **What does `shift()` do?**
    - `shift()` moves values in a Series or DataFrame along an index by a specified number of periods, facilitating time-series lag comparisons, differences, and growth calculations.

39. **What is a rolling average?**
    - A rolling average (moving average) calculates the mean of data points within a sliding time window (e.g. 7 days, 30 days) to smooth out short-term fluctuations and reveal underlying trend trajectories.

40. **Why should EDA be performed before machine learning?**
    - EDA ensures the modeler understands data types, distributions, missing value mechanisms, outliers, multicollinearity, and class balances, directly guiding feature engineering, imputation strategies, and model selection.
