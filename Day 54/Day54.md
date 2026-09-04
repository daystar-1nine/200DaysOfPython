# Day 54 — NumPy Fundamentals & Numerical Data Processing in Python

---

## 📚 Masterclass Overview

Today marks the major transition point in your 200-day Python journey: transitioning from general-purpose Python backend engineering into the **Data Science, Data Engineering, and Machine Learning ecosystem**.

The foundational bedrock of scientific computing and data analysis in Python is **NumPy** (**Num**erical **Py**thon). NumPy introduces the `ndarray` (N-Dimensional Array), a contiguous block of homogeneous memory optimized in C that enables high-performance matrix algebra, multidimensional tensor operations, and vectorized computations operating orders of magnitude faster than standard Python loops.

---

## 🧠 Core Concepts & Technical Architecture

### 1. Why NumPy? (NumPy `ndarray` vs. Python `list`)

Standard Python lists are arrays of pointers to heterogeneous Python objects scattered across memory. Each list element incurs dynamic typing overhead, reference counting, and garbage collection checks during iteration.

NumPy's `ndarray`:
- Stores elements in **contiguous memory blocks**.
- Enforces **homogeneous data types** (`float64`, `int32`, etc.).
- Executes vectorized loops in compiled C/Fortran code without GIL overhead.

| Attribute | Standard Python `list` | NumPy `ndarray` |
| :--- | :--- | :--- |
| **Memory Allocation** | Non-contiguous (Pointers to PyObjects) | Contiguous unboxed C memory block |
| **Element Types** | Heterogeneous (Any PyObject) | Homogeneous (Single uniform `dtype`) |
| **Operation Speed** | Slow interpreted `for` loops | Fast vectorized SIMD C execution |
| **Memory Overhead** | High (28+ bytes per integer object) | Low (8 bytes for `int64` / `float64`) |

---

### 2. Array Attributes: `ndim`, `shape`, `size`, `dtype`

Every NumPy `ndarray` instance possesses key structural metadata properties:

```python
import numpy as np

matrix = np.array([
    [80, 85, 90],
    [70, 75, 80]
])

print(matrix.ndim)   # 2  -> Number of dimensions (axes)
print(matrix.shape)  # (2, 3) -> Dimension sizes (2 rows, 3 columns)
print(matrix.size)   # 6  -> Total number of elements (2 * 3)
print(matrix.dtype)  # int64 -> Uniform data type
```

---

### 3. Array Indexing & Slicing

- **1D Slicing**: `array[start:stop:step]`
- **2D Indexing**: `matrix[row_idx, col_idx]`
- **2D Slicing**: `matrix[row_start:row_stop, col_start:col_stop]`
- Using `:` along an axis selects all elements across that dimension.

```python
matrix = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print(matrix[0, 2])    # 30 (Row 0, Column 2)
print(matrix[:2, :2])  # Top-left 2x2 sub-matrix [[10, 20], [40, 50]]
print(matrix[:, 1])    # Entire 2nd column [20, 50, 80]
```

---

### 4. Vectorization & Scalar Arithmetic

Vectorization performs element-wise operations across entire arrays in C speed without explicit Python `for` loops:

```python
marks = np.array([75, 80, 85])
bonus = marks + 5          # [80, 85, 90]  (Scalar addition)
scaled = marks * 1.1       # [82.5, 88.0, 93.5]
```

---

### 5. Broadcasting Rules

Broadcasting allows NumPy to perform arithmetic operations on arrays of different shapes automatically without making redundant memory copies.

**Broadcasting Rule**: Two dimensions are compatible when:
1. They are equal, OR
2. One of them is 1.

```python
# 2D Array (2x3) + 1D Array (1x3 or 3,)
marks = np.array([
    [80, 90, 70],
    [60, 75, 85]
])
weights = np.array([1.1, 1.0, 1.2])

weighted_marks = marks * weights
# Result: [[88.0, 90.0, 84.0], [66.0, 75.0, 102.0]]
```

---

### 6. Boolean Masking & Logical Indexing

Boolean masks filter array elements conditionally using bitwise operators (`&` AND, `|` OR, `~` NOT):

```python
marks = np.array([45, 67, 89, 32, 95])
mask = (marks >= 60) & (marks <= 90)
filtered = marks[mask]  # Output: [67, 89]
```

---

### 7. Understanding NumPy Axes (`axis=0` vs `axis=1`)

Aggregations (`sum`, `mean`, `std`, `min`, `max`, `argmax`, `argmin`) operate across specified axes:
- `axis=0`: Collapse down rows $\rightarrow$ Calculate **column-wise** statistics.
- `axis=1`: Collapse across columns $\rightarrow$ Calculate **row-wise** statistics.

```python
marks = np.array([
    [80, 90, 70],  # Student 1
    [60, 75, 85],  # Student 2
    [95, 88, 92]   # Student 3
])

student_averages = np.mean(marks, axis=1)  # [80.0, 73.33, 91.67] (Row-wise)
subject_averages = np.mean(marks, axis=0)  # [78.33, 84.33, 82.33] (Column-wise)
```

---

### 8. Array Creation & Reshaping Helpers

- `np.arange(start, stop, step)`: Creates evenly spaced sequences.
- `np.linspace(start, stop, num)`: Creates `num` linearly spaced values over `[start, stop]`.
- `np.zeros(shape)`, `np.ones(shape)`, `np.full(shape, fill_value)`, `np.eye(N)`: Pre-allocated arrays.
- `array.reshape(rows, cols)`: Change array dimensions (must preserve total `size`).
- `array.reshape(rows, -1)`: Automatically infers missing dimension.
- `array.T`: Transpose matrix (swaps rows and columns).

---

### 9. Min-Max Score Normalization Formula

Min-Max normalization rescales continuous values to the range $[0.0, 1.0]$:

$$\text{Normalized}(x) = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

```python
scores = np.array([10, 20, 30, 40, 50])
normalized = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
# Output: [0.00, 0.25, 0.50, 0.75, 1.00]
```

---

## ❓ 25 Technical Interview Questions & Answers

### Q1: What is NumPy and why is it essential for Python Data Science?
**Answer:** NumPy (Numerical Python) provides the N-dimensional array object (`ndarray`) and vectorized mathematical operations implemented in compiled C. It serves as the core computational engine behind Pandas, SciPy, Matplotlib, scikit-learn, PyTorch, and TensorFlow.

### Q2: Why is a NumPy `ndarray` faster than a standard Python `list`?
**Answer:** A Python `list` contains pointers to dispersed heap objects requiring dynamic type checking and unboxing overhead during iterations. A NumPy `ndarray` stores homogeneous data in contiguous memory, enabling C-level SIMD (Single Instruction, Multiple Data) CPU vectorization and cache locality.

### Q3: What is the difference between `ndim`, `shape`, `size`, and `dtype`?
**Answer:**
- `ndim`: Integer count of array dimensions/axes (e.g. 1 for vector, 2 for matrix).
- `shape`: Tuple specifying the number of elements along each dimension (e.g. `(3, 4)`).
- `size`: Total count of elements in the array (product of shape dimensions).
- `dtype`: Data type descriptor of the homogeneous array elements (e.g. `float64`, `int32`).

### Q4: What does the term "vectorization" mean in NumPy?
**Answer:** Vectorization refers to executing element-wise mathematical operations on entire array structures simultaneously in C without writing explicit interpreted Python `for` loops.

### Q5: What is broadcasting in NumPy?
**Answer:** Broadcasting is the set of rules by which NumPy performs arithmetic operations on arrays of different shapes by implicitly expanding the smaller array along dimensions of size 1 without copying memory.

### Q6: What are the two main conditions for two array dimensions to be compatible for broadcasting?
**Answer:** Two dimensions are compatible when:
1. They are equal, OR
2. One of the dimensions is equal to 1.

### Q7: How does slicing a NumPy array differ from slicing a standard Python list?
**Answer:** Slicing a Python list creates a new copy of the data elements. Slicing a NumPy array returns a **view** of the original memory buffer; modifying slice elements mutates the underlying original array unless `.copy()` is explicitly called.

### Q8: Explain the difference between `axis=0` and `axis=1` in 2D array aggregations.
**Answer:**
- `axis=0`: Collapses down rows, performing operations along columns (column-wise results).
- `axis=1`: Collapses across columns, performing operations along rows (row-wise results).

### Q9: What is the difference between `np.arange()` and `np.linspace()`?
**Answer:**
- `np.arange(start, stop, step)`: Generates numbers by specifying a fixed step size increment.
- `np.linspace(start, stop, num)`: Generates `num` evenly spaced sample numbers over a specified closed interval.

### Q10: Why must you use `&`, `|`, and `~` instead of `and`, `or`, and `not` for NumPy Boolean masks?
**Answer:** `and`, `or`, and `not` evaluate truthiness of an entire object as a whole (raising `ValueError: Truth value of an array is ambiguous`). Bitwise operators (`&`, `|`, `~`) evaluate element-wise boolean operations across ndarrays.

### Q11: What does `np.argmax()` and `np.argmin()` return?
**Answer:** `np.argmax()` returns the index position of the maximum value in an array (or along a specified axis). `np.argmin()` returns the index position of the minimum value.

### Q12: What does `np.argsort()` return?
**Answer:** `np.argsort()` returns an array of indices that would sort the original array along the specified axis.

### Q13: What happens if you attempt to call `.reshape(3, 4)` on an array of 10 elements?
**Answer:** NumPy raises a `ValueError: cannot reshape array of size 10 into shape (3,4)` because the total element count (`size = 10`) must equal the product of the target shape dimensions ($3 \times 4 = 12$).

### Q14: What is the purpose of passing `-1` in `array.reshape(3, -1)`?
**Answer:** `-1` acts as a placeholder telling NumPy to automatically compute and infer the size of that specific dimension based on the total array size and other specified shape dimensions.

### Q15: What is a Boolean mask array?
**Answer:** A Boolean mask array is an `ndarray` of `dtype=bool` generated by applying comparison operators (e.g. `arr > 50`). Passing this mask into indexing brackets `arr[mask]` filters and extracts only the elements corresponding to `True`.

### Q16: How do you create an identity matrix in NumPy?
**Answer:** Using `np.eye(N)`, which returns a 2D square matrix of shape $(N, N)$ with ones on the main diagonal and zeros elsewhere.

### Q17: What is the difference between `np.zeros((3,3))` and `np.ones((3,3))`?
**Answer:** `np.zeros((3,3))` creates a $3 \times 3$ matrix initialized entirely with floating-point `0.0`, while `np.ones((3,3))` initializes all elements with floating-point `1.0`.

### Q18: How do you transpose a 2D matrix in NumPy?
**Answer:** By accessing the `.T` attribute property (e.g. `matrix.T`), which swaps the rows and columns of the 2D array.

### Q19: What is standard deviation and how do you calculate it in NumPy?
**Answer:** Standard deviation measures the spread or dispersion of data points relative to their mean. In NumPy, it is computed using `np.std(array)`.

### Q20: How do you generate reproducible random numbers in modern NumPy?
**Answer:** By instantiating a random number generator object with a fixed integer seed: `rng = np.random.default_rng(seed=42)`.

### Q21: What is the Min-Max normalization formula?
**Answer:** $\text{Normalized}(x) = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$, mapping raw numeric scores to the closed range $[0.0, 1.0]$.

### Q22: What does `np.full((2, 3), 7)` produce?
**Answer:** It produces a $2 \times 3$ matrix where every element is filled with the integer value `7`.

### Q23: How do you change the data type of an existing NumPy array?
**Answer:** By calling the `.astype()` method (e.g., `float_arr = int_arr.astype(np.float64)`).

### Q24: What is the memory complexity advantage of using NumPy vectorization over pure Python loops?
**Answer:** Vectorization executes operations directly in unboxed contiguous C arrays without instantiating intermediate Python integer/float objects, reducing memory overhead and eliminating interpreter loop dispatch time.

### Q25: How will NumPy concepts transfer to Pandas in upcoming lessons?
**Answer:** Pandas `Series` and `DataFrame` objects are built directly on top of NumPy `ndarray` structures. Pandas indexing, filtering, element-wise math, axis aggregations, and broadcasting draw directly upon NumPy's underlying architecture.
