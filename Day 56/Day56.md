# DAY 56 — PANDAS FUNDAMENTALS: SERIES & DATAFRAMES

---

## 🚀 Module Overview

Welcome to Day 56 of the 200 Days of Python Challenge. Today marks your official entrance into **Pandas**, Python's premier library for tabular data manipulation and analytical processing. 

While NumPy provides low-level multi-dimensional array operations, **Pandas** introduces labelled, heterogeneous data structures (`Series` and `DataFrame`) designed for real-world business datasets, CSV/SQL/JSON operations, missing value handling, alignment, filtering, and aggregation.

---

## 📌 Masterclass Theory Notes

### 1. What is Pandas & Why Do We Need It?
- **Definition**: Pandas is built on top of NumPy, offering fast, flexible data structures (`Series` and `DataFrame`) for relational or labelled data.
- **NumPy vs Pandas**:
  - **NumPy**: Homogeneous arrays, numerical computations, linear algebra, performance-optimized vectorization.
  - **Pandas**: Heterogeneous tables, column names, row indices, time-series, missing data handling, file I/O (CSV, Excel, Parquet, SQL).

---

### 2. Core Data Structures: Series & DataFrame

#### A. Pandas Series (1D Labelled Array)
A `Series` is a one-dimensional array-like object containing an array of data and an associated array of data labels called its `index`.

```python
import pandas as pd

# Creating a Series
marks = pd.Series([85, 92, 78, 90], index=['Alice', 'Bob', 'Charlie', 'David'], name="Math_Score")
print(marks)
# Output:
# Alice      85
# Bob        92
# Charlie    78
# David      90
# Name: Math_Score, dtype: int64
```

#### B. Pandas DataFrame (2D Labelled Table)
A `DataFrame` represents a tabular, spreadsheet-like data structure containing an ordered collection of columns, each of which can be a different value type (numeric, string, boolean, datetime).

```python
data = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Math": [85, 92, 78, 90],
    "Physics": [88, 79, 95, 91]
}
df = pd.DataFrame(data)
print(df)
```

---

### 3. Data Reading & Writing (I/O)

| Operation | Pandas Method | Example Usage |
|---|---|---|
| Read CSV | `pd.read_csv()` | `df = pd.read_csv('students.csv', index_col=0)` |
| Write CSV | `df.to_csv()` | `df.to_csv('output.csv', index=False)` |
| Read Excel | `pd.read_excel()` | `df = pd.read_excel('sales.xlsx', sheet_name='Sheet1')` |
| Write Excel | `df.to_excel()` | `df.to_excel('export.xlsx', index=False)` |
| Read JSON | `pd.read_json()` | `df = pd.read_json('data.json')` |

---

### 4. Indexing & Selection (`loc` vs `iloc`)

- **`.loc[]`**: Label-based indexing. Accepts row/column index names.
- **`.iloc[]`**: Integer-position based indexing. Accepts 0-indexed row/column numbers.

```python
# Label indexing: Row 'Alice', Column 'Math'
score = df.loc[0, 'Math']

# Integer position indexing: Row 0, Column 1
score_pos = df.iloc[0, 1]

# Slicing with loc (inclusive of stop boundary!)
df.loc[0:2, 'Name':'Math']

# Slicing with iloc (exclusive of stop boundary!)
df.iloc[0:2, 0:2]
```

---

### 5. Filtering Data (Boolean Masking)

Always use vectorized bitwise operators (`&` for AND, `|` for OR, `~` for NOT) when evaluating multiple conditions on Pandas Series/DataFrames. Always wrap conditions in parentheses `()`.

```python
# Filtering students in CSE department with Math score >= 80
cse_top = df[(df['Department'] == 'CSE') & (df['Math'] >= 80)]

# Filtering students who scored above 90 in Physics OR Math
high_scorers = df[(df['Math'] > 90) | (df['Physics'] > 90)]
```

---

### 6. Data Cleaning & Missing Values

- **Identify missing**: `df.isnull()` or `df.isna()`
- **Count missing**: `df.isnull().sum()`
- **Fill missing**: `df.fillna({'Math': df['Math'].mean()})`
- **Drop missing**: `df.dropna(subset=['Math'])`
- **Drop duplicates**: `df.drop_duplicates(subset=['Student_ID'], keep='first')`

---

### 7. Sorting & Calculated Columns

```python
# Sorting values descending by Total Marks
df_sorted = df.sort_values(by='Total', ascending=False)

# Adding derived calculated columns
df['Total'] = df['Math'] + df['Physics'] + df['Chemistry']
df['Average'] = df['Total'] / 3.0

# Using cut for grade categorization
df['Grade'] = pd.cut(
    df['Average'], 
    bins=[0, 50, 70, 85, 100], 
    labels=['F', 'C', 'B', 'A']
)
```

---

### 8. Aggregations & Summary Statistics

- `df.describe()`: Provides count, mean, std, min, 25%, 50%, 75%, max.
- `df['Department'].value_counts()`: Frequency table of unique values.
- `df['Math'].idxmax()`: Index label of maximum value.
- `df['Math'].mean()`: Column mean ignoring NaNs.

---

## ❓ 25 Technical Interview Questions & Answers

### Q1: What is the fundamental difference between NumPy arrays and Pandas DataFrames?
**Answer**: NumPy arrays are homogeneous multi-dimensional data blocks designed for numerical and array-based linear algebra operations. Pandas DataFrames are 2D heterogeneous tabular structures with row and column labels, designed for relational data manipulation, missing value handling, alignment, and dataset merging/reshaping.

### Q2: What is the difference between `loc` and `iloc` in Pandas?
**Answer**: `loc` relies on index labels (e.g. `df.loc['row_name', 'col_name']`) and includes the stop boundary when slicing. `iloc` relies strictly on integer positional indexing (e.g. `df.iloc[0:5, 1:3]`) and excludes the stop boundary (standard Python slice behavior).

### Q3: Why must we use bitwise operators (`&`, `|`, `~`) instead of logical keywords (`and`, `or`, `not`) for Pandas filtering?
**Answer**: Python's `and`/`or` evaluate truthiness for an entire object as a single boolean value, which raises a `ValueError: The truth value of a Series is ambiguous` in Pandas. Bitwise operators (`&`, `|`, `~`) perform element-wise boolean operations across entire Series vectors.

### Q4: How does Pandas handle missing data (`NaN` or `None`) during aggregation functions like `.sum()` or `.mean()`?
**Answer**: By default, Pandas aggregate functions automatically skip missing values (`skipna=True`). For example, `df['marks'].mean()` computes the average over only non-null values.

### Q5: How do you read a CSV file into a DataFrame while specifying custom missing value tokens?
**Answer**: Use the `na_values` argument in `pd.read_csv()`:
`df = pd.read_csv('data.csv', na_values=['NA', 'N/A', 'missing', '-999', ''])`.

### Q6: What is the difference between `.copy()` and assigning a DataFrame slice to a new variable?
**Answer**: Assigning a DataFrame slice creates a view or reference to the original buffer. Modifying it can cause a `SettingWithCopyWarning` or alter the underlying parent DataFrame unexpectedly. `.copy()` creates a new independent object in memory with deep data copying.

### Q7: How can you modify column names of a DataFrame?
**Answer**: Either assign a new list to `df.columns` (e.g., `df.columns = ['A', 'B', 'C']`) or use `df.rename(columns={'old_name': 'new_name'}, inplace=True)`.

### Q8: What does `inplace=True` do in Pandas operations, and why is its usage controversial in modern Pandas?
**Answer**: `inplace=True` attempts to modify the DataFrame object in place without returning a new object. It is controversial because it often does not provide memory/performance optimization under the hood and disrupts method chaining. Modern Pandas best practices recommend reassigning: `df = df.drop(columns=['col'])`.

### Q9: How do you identify duplicate rows based on specific key columns in Pandas?
**Answer**: Use `df.duplicated(subset=['ID'], keep='first')` to return a boolean mask of duplicate rows, or `df.drop_duplicates(subset=['ID'], keep='first')` to drop duplicate rows directly.

### Q10: What is `pd.cut()` and how does it differ from `pd.qcut()`?
**Answer**: `pd.cut()` bins data into discrete intervals based on numerical edge values (equal-width bins). `pd.qcut()` bins data based on sample quantiles, ensuring each bin contains roughly an equal number of observations (equal-frequency bins).

### Q11: How do you sort a DataFrame by multiple columns with different sort orders?
**Answer**: Pass lists to `by` and `ascending` parameters: `df.sort_values(by=['Department', 'Score'], ascending=[True, False])`.

### Q12: What does `df.describe()` return for numeric vs categorical columns?
**Answer**: For numeric columns, it returns count, mean, std, min, 25%, 50%, 75%, and max. For object/categorical columns, it returns count, unique, top (most frequent value), and freq (frequency of top value).

### Q13: How can you convert a column's data type in Pandas?
**Answer**: Use `df['col'] = df['col'].astype(float)` or `pd.to_numeric(df['col'], errors='coerce')` to parse numerical strings and turn unparseable values into `NaN`.

### Q14: What is the purpose of `reset_index()`?
**Answer**: `reset_index()` resets the row index of a DataFrame back to the default integer index (`0, 1, 2...`). Setting `drop=True` avoids keeping the old index as a column in the DataFrame.

### Q15: How can you filter rows where a string column contains a specific substring?
**Answer**: Use the vectorized string accessor `.str.contains()`:
`df[df['Name'].str.contains('John', case=False, na=False)]`.

### Q16: How do you count unique values in a Pandas Series?
**Answer**: Use `series.value_counts()` to return a frequency table, or `series.nunique()` to return the integer count of distinct non-null values.

### Q17: What happens when you perform vector addition on two Series with non-matching index labels?
**Answer**: Pandas automatically aligns the Series by their index labels. Any index label present in one Series but not the other results in `NaN` at that index position unless `series.add(series2, fill_value=0)` is used.

### Q18: How do you find the index label of the row with the maximum value in a column?
**Answer**: Use `df['column'].idxmax()`. To find the integer position instead, use `df['column'].to_numpy().argmax()`.

### Q19: How can you drop columns or rows from a DataFrame?
**Answer**: Use `df.drop(columns=['col1', 'col2'])` to drop columns, or `df.drop(index=[0, 1])` / `df.drop([0, 1], axis=0)` to drop rows.

### Q20: How do you transpose a DataFrame (swap rows and columns)?
**Answer**: Use `df.T` or `df.transpose()`.

### Q21: What is the function of `df.info()`?
**Answer**: `df.info()` prints a concise summary of the DataFrame including index dtype, memory usage, number of non-null values per column, and column data types.

### Q22: How can you set a specific column as the DataFrame index?
**Answer**: Use `df.set_index('Student_ID', inplace=True)`.

### Q23: How do you apply a custom Python function row-by-row or element-by-element across a DataFrame?
**Answer**: Use `df['col'].apply(custom_func)` for a single Series or `df.apply(custom_func, axis=1)` for row-wise evaluation across multiple columns.

### Q24: What is the difference between `isna()` and `isnull()` in Pandas?
**Answer**: They are exact aliases of each other. `isnull()` is inherited from R compatibility, whereas `isna()` aligns with NumPy's `isnan()`.

### Q25: How do you write a DataFrame to a CSV file without including the row index?
**Answer**: Set `index=False` when calling `to_csv()`: `df.to_csv('output.csv', index=False)`.
