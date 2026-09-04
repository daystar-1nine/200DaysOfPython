# Day 55 — Advanced NumPy: Indexing, Broadcasting, Views, NaN Handling & Linear Algebra

---

## 📚 Masterclass Overview

Following yesterday's NumPy fundamentals introduction, today's masterclass elevates your numerical data engineering capabilities to advanced level. In real-world data science, numerical datasets are often incomplete, multidimensional, unstructured, or require matrix operations. 

Today we master:
1. **Fancy & Integer-Array Indexing**: Non-contiguous element extraction across dimensions.
2. **Conditional Replacement & `np.where()`**: In-place filtering and element-wise `if-else` branching.
3. **Sorting & Frequency Tallying**: `np.sort()`, `np.argsort()`, `np.argmax()`, `np.argmin()`, and `np.unique()`.
4. **Array Manipulation & Stacking**: `np.concatenate()`, `np.vstack()`, `np.hstack()`, `np.split()`, and `np.array_split()`.
5. **Memory Views vs. Deep Copies**: Understanding shared data buffers to prevent unintended mutation side-effects.
6. **In-Depth Broadcasting**: Multi-row arithmetic broadcasting across mismatched dimensions.
7. **NaN & Missing Value Mechanics**: `np.nan`, `np.isnan()`, `np.nanmean()`, `np.nansum()`, and NaN imputation strategies.
8. **Linear Algebra Foundations**: Dot products (`np.dot`), matrix multiplication (`A @ B`, `np.matmul`), and transpose (`.T`).
9. **Multi-Condition Categorization**: Implementing vectorized grading systems using `np.select()`.

---

## 🧠 Core Concepts & Technical Architecture

### 1. Fancy / Integer-Array Indexing
Fancy indexing passes an array or list of integer indices to select non-contiguous elements from specific position coordinates:

```python
import numpy as np

# 1D Fancy Indexing
numbers = np.array([10, 20, 30, 40, 50])
selected = numbers[[0, 2, 4]]  # [10, 30, 50]

# 2D Fancy Indexing (Selecting specific coordinate pairs)
matrix = np.array([
    [80, 90, 70],
    [60, 75, 85],
    [95, 88, 92]
])
# Select (row 0, col 1) -> 90, (row 1, col 2) -> 85, (row 2, col 0) -> 95
coords = matrix[[0, 1, 2], [1, 2, 0]]  # [90, 85, 95]
```

---

### 2. Conditional Replacement & `np.where()`

- **In-place Replacement**:
  ```python
  marks = np.array([35, 45, 78, 29, 90])
  marks[marks < 40] = 40  # [40, 45, 78, 40, 90]
  ```

- **Functional Branching (`np.where(condition, x, y)`)**:
  ```python
  status = np.where(marks >= 40, "Pass", "Fail")
  # ['Pass', 'Pass', 'Pass', 'Pass', 'Pass']
  ```

---

### 3. Sorting & Ranking (`sort`, `argsort`, `unique`)

- `np.sort(arr)`: Returns a new array containing sorted values.
- `np.argsort(arr)`: Returns the array of **index positions** that would sort the array.
- `np.argsort(arr)[::-1]`: Returns descending order indices for ranking.
- `np.unique(arr, return_counts=True)`: Extracts unique elements and their frequency counts.

```python
courses = np.array(["DS", "CS", "DS", "AI", "CS", "DS"])
vals, counts = np.unique(courses, return_counts=True)
# vals: ['AI', 'CS', 'DS'], counts: [1, 2, 3]
```

---

### 4. Multi-Condition Categorization with `np.select()`

`np.select(condlist, choicelist, default)` evaluates a list of boolean conditions in sequence and assigns matching choices:

```python
averages = np.array([92.5, 84.0, 76.0, 62.0, 45.0, 32.0])

conditions = [
    averages >= 90,
    averages >= 80,
    averages >= 70,
    averages >= 60,
    averages >= 50,
    averages >= 40
]
choices = ["A+", "A", "B", "C", "D", "E"]

grades = np.select(conditions, choices, default="F")
# ['A+', 'A', 'B', 'C', 'E', 'F']
```

---

### 5. Memory Management: Views vs. Deep Copies

- **View**: A slice of a NumPy array (e.g. `b = a[1:3]`) returns a **view** sharing the underlying memory buffer of `a`. Modifying `b` mutates `a`!
- **Copy**: `b = a[1:3].copy()` allocates a completely independent memory buffer.

```python
a = np.array([10, 20, 30, 40])
view_b = a[1:3]
view_b[0] = 999  # Modifies original array 'a' -> [10, 999, 30, 40]

copy_c = a[1:3].copy()
copy_c[0] = 555  # 'a' remains unchanged
```

---

### 6. Array Stacking & Splitting

- `np.vstack((a, b))`: Stacks arrays vertically along rows ($axis=0$).
- `np.hstack((a, b))`: Stacks arrays horizontally along columns ($axis=1$).
- `np.concatenate((a, b), axis=0)`: General concatenation.
- `np.split(arr, N)`: Splits array into $N$ equal sections (raises `ValueError` if size is not divisible by $N$).
- `np.array_split(arr, N)`: Splits array into $N$ sections, tolerating unequal section sizes.

---

### 7. Missing Value Mechanics (`NaN` & `np.isnan`)

- `np.nan`: Floating-point "Not a Number" constant representing missing data.
- **Propagation Problem**: Standard aggregate functions (`np.mean`, `np.sum`) return `nan` if any element is `nan`.
- **Solution**: Use `np.isnan(arr)` for boolean masking or NaN-aware functions:
  - `np.nanmean()`, `np.nansum()`, `np.nanmin()`, `np.nanmax()`, `np.nanmedian()`, `np.nanargmax()`, `np.nanargmin()`.

```python
marks = np.array([80.0, 90.0, np.nan, 70.0])

print(np.mean(marks))     # nan (Standard mean propagates NaN)
print(np.nanmean(marks))  # 80.0 (NaN-aware mean ignores missing values)

# Replace NaNs with column mean:
mask = np.isnan(marks)
marks[mask] = np.nanmean(marks)  # [80.0, 90.0, 80.0, 70.0]
```

---

### 8. Basic Linear Algebra & Matrix Operations

- **Dot Product**: `np.dot(a, b)` calculates inner product $\sum a_i b_i$.
- **Matrix Multiplication**: `A @ B` or `np.matmul(A, B)` computes matrix product $(M \times K) \cdot (K \times N) \rightarrow (M \times N)$.
- **Transpose**: `A.T` swaps matrix rows and columns.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = A @ B
# [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]] -> [[19, 22], [43, 50]]
```

---

## ❓ 25 Technical Interview Questions & Answers

### Q1: What is fancy indexing in NumPy?
**Answer:** Fancy indexing (or integer-array indexing) refers to passing an array or list of integers to select non-contiguous elements from specific position coordinates of an `ndarray`.

### Q2: What is the key functional difference between `np.sort()` and `np.argsort()`?
**Answer:** `np.sort()` returns a sorted copy of the array elements. `np.argsort()` returns an array of integer index positions that would sort the original array, enabling multi-column or secondary metric ranking.

### Q3: How do you rank an array in descending order using `np.argsort()`?
**Answer:** By applying slice reversal to the argsort output array: `rankings = np.argsort(arr)[::-1]`.

### Q4: What does `np.unique(arr, return_counts=True)` return?
**Answer:** It returns a tuple of two 1D arrays: the sorted unique elements of `arr`, and an integer array containing the frequency count of each unique element.

### Q5: What is the difference between a NumPy View and a NumPy Copy?
**Answer:** A view is a shallow reference sharing the underlying memory buffer of the original array (mutations affect the source). A copy is an independent memory buffer allocation created via `.copy()`.

### Q6: Why does slicing a NumPy array return a View instead of a Copy by default?
**Answer:** Returning views avoids copying memory buffers when manipulating large multi-gigabyte matrices, optimizing memory usage and execution speed.

### Q7: What is `np.where(condition, x, y)` and how does it work?
**Answer:** `np.where` is a vectorized ternary operator. It evaluates `condition` element-wise, returning values from `x` where `True` and from `y` where `False`.

### Q8: How does `np.select(condlist, choicelist, default)` handle multi-condition branching?
**Answer:** `np.select` evaluates a list of boolean conditions in order. For each element, it picks the corresponding choice from `choicelist` for the first condition that evaluates to `True`, falling back to `default` if all fail.

### Q9: What is `np.nan` and what data type is required to store `np.nan` in an array?
**Answer:** `np.nan` represents "Not a Number" missing values. Because `np.nan` is an IEEE 754 floating-point standard, an array must have a floating-point `dtype` (`float32` or `float64`) to hold `np.nan`.

### Q10: Why does `np.mean()` return `nan` when applied to an array containing `np.nan` values?
**Answer:** Standard mathematical aggregations propagate missing values because arithmetic involving `nan` (e.g. $80 + \text{nan}$) yields `nan`.

### Q11: How do NaN-aware functions (e.g. `np.nanmean()`) handle missing values?
**Answer:** NaN-aware functions filter out `np.nan` values before executing the aggregation, computing metrics only over valid numerical elements.

### Q12: How do you replace all `np.nan` values in an array with the array's mean value?
**Answer:** Calculate the NaN-aware mean `mean_val = np.nanmean(arr)` and apply boolean mask assignment: `arr[np.isnan(arr)] = mean_val`.

### Q13: What is the difference between `np.vstack()` and `np.hstack()`?
**Answer:** `np.vstack()` stacks arrays vertically along rows ($axis=0$). `np.hstack()` stacks arrays horizontally side-by-side along columns ($axis=1$).

### Q14: What is the difference between `np.split()` and `np.array_split()`?
**Answer:** `np.split()` raises a `ValueError` if the array size cannot be divided equally into the requested number of sub-arrays. `np.array_split()` accepts unequal section sizes gracefully.

### Q15: How does broadcasting operate when adding a 1D array of shape `(3,)` to a 2D array of shape `(2, 3)`?
**Answer:** NumPy stretches the 1D array along axis 0, adding the 3-element vector to every row of the 2D matrix.

### Q16: What operator is used for matrix multiplication in Python 3.5+ for NumPy arrays?
**Answer:** The `@` matrix multiplication operator (e.g. `C = A @ B`), which calls `np.matmul()`.

### Q17: What are the dimension rules for multiplying two matrices $A$ ($M \times K$) and $B$ ($P \times N$)?
**Answer:** The inner dimensions must match ($K = P$). The resulting product matrix has shape $(M \times N)$.

### Q18: What is the dot product of two 1D vectors $a$ and $b$?
**Answer:** The sum of the element-wise products of the vectors: $\text{dot}(a, b) = \sum_{i=1}^n a_i b_i$.

### Q19: What does `np.argmax()` return on a 2D matrix when `axis=1` is specified?
**Answer:** It returns a 1D array containing the column index of the maximum value for each row.

### Q20: What does `np.argmin()` return on a 2D matrix when `axis=0` is specified?
**Answer:** It returns a 1D array containing the row index of the minimum value for each column.

### Q21: How do you conditionally replace negative numbers in an array with `0` without creating a new array?
**Answer:** Use boolean indexing assignment: `arr[arr < 0] = 0`.

### Q22: Why is reproducible random number generation important in computational science and data analytics?
**Answer:** Using a fixed seed (`np.random.default_rng(seed=42)`) ensures that synthetic data, train/test splits, and stochastic experiments produce identical results across runs, enabling verification and debugging.

### Q23: How do you extract rows from a 2D matrix where the row average is greater than 75?
**Answer:** Calculate row averages `row_avgs = np.mean(matrix, axis=1)`. Filter using boolean mask: `filtered_matrix = matrix[row_avgs > 75]`.

### Q24: What is the output of `np.unique()` on a 2D matrix?
**Answer:** By default, `np.unique()` flattens the 2D matrix into a 1D array before returning the sorted unique elements.

### Q25: How do NumPy advanced indexing and broadcasting concepts transition to Pandas?
**Answer:** Pandas DataFrames utilize NumPy's advanced indexing, views vs copies logic, NaN handling (`dropna`, `fillna`), and broadcasting engine under the hood for vectorized tabular manipulation.
