# DAY 57 — PANDAS DATA MANIPULATION & AGGREGATION

---

## 🚀 Module Overview

Welcome to Day 57 of the 200 Days of Python Challenge. Yesterday you mastered the fundamentals of Pandas: Series, DataFrames, Data I/O, Indexing (`loc`, `iloc`), Filtering, and basic descriptive statistics.

Today, we dive into the core engine of real-world data analytics: **Data Manipulation, Merging, Reshaping, and Aggregation**. 

By combining rows, grouping categories, performing multi-metric aggregations, merging relational datasets, and computing pivot tables, you can transform millions of raw transactional rows into actionable business intelligence.

---

## 📌 Masterclass Theory Notes

### 1. Grouping Data with `groupby()`
The `groupby()` method implements the classic **Split-Apply-Combine** strategy:
1. **Split**: Segregate data into groups based on key column(s).
2. **Apply**: Compute an aggregation function (e.g., `sum()`, `mean()`, `count()`) independently on each group.
3. **Combine**: Merge the aggregated results back into a single Series or DataFrame.

```python
import pandas as pd

df = pd.DataFrame({
    "Region": ["West", "East", "West", "South"],
    "Sales": [800, 500, 600, 900]
})

# Total Sales by Region
region_sales = df.groupby("Region")["Sales"].sum()
print(region_sales)
```

---

### 2. Multi-Metric Aggregations & Named Aggregations

#### A. Multiple Aggregations using `.agg()`
Pass a list of functions to `.agg()` to compute multiple metrics simultaneously:
```python
df.groupby("Region")["Sales"].agg(["sum", "mean", "min", "max", "count"])
```

#### B. Named Aggregations (Professional Technique)
Specify explicit custom column names and target functions cleanly:
```python
df.groupby("Region").agg(
    total_sales=("Sales", "sum"),
    average_sales=("Sales", "mean"),
    highest_sale=("Sales", "max"),
    order_count=("Sales", "count")
)
```

---

### 3. Grouping by Multiple Columns & `reset_index()`
Group across multiple hierarchical levels (e.g., `Region` and `Category`):
```python
multi_group = df.groupby(["Region", "Category"])["Sales"].sum().reset_index()
```
`reset_index()` converts index labels back into standard DataFrame columns, which is essential for downstream processing and exports.

---

### 4. Merging Datasets (`pd.merge()` vs `.join()`)

Pandas `pd.merge()` mirrors SQL `JOIN` syntax across relational tables:

| Join Type (`how`) | SQL Equivalent | Description |
|---|---|---|
| `inner` (default) | `INNER JOIN` | Keeps rows where keys exist in **both** DataFrames. |
| `left` | `LEFT JOIN` | Keeps **all** rows from left DataFrame, filling missing right values with `NaN`. |
| `right` | `RIGHT JOIN` | Keeps **all** rows from right DataFrame, filling missing left values with `NaN`. |
| `outer` | `FULL OUTER JOIN` | Keeps **all** rows from both DataFrames, filling missing entries with `NaN`. |

```python
# Inner Merge on Customer_ID key
merged_df = pd.merge(customers_df, orders_df, on="Customer_ID", how="inner")
```

> **Difference between `merge()` and `join()`**:
> - `pd.merge()` joins DataFrames primarily based on **column values**.
> - `df.join()` joins DataFrames primarily based on **index labels**.

---

### 5. Concatenating DataFrames (`pd.concat()`)
Use `pd.concat()` to stack DataFrames vertically (row-wise, `axis=0`) or align horizontally (column-wise, `axis=1`).

```python
# Stack rows (axis=0) and reset index sequence
combined_df = pd.concat([jan_sales, feb_sales], axis=0, ignore_index=True)
```

---

### 6. Element Transformations: `apply()`, `map()`, and `replace()`

- **`map()`**: Element-wise transformation on a **Series** using a dictionary mapping or function.
  ```python
  df["Dept_Full"] = df["Dept"].map({"CSE": "Computer Science", "DS": "Data Science"})
  ```
- **`replace()`**: Replaces specific values across a **Series or DataFrame**.
  ```python
  df["Status"] = df["Status"].replace({"P": "Pass", "F": "Fail"})
  ```
- **`apply()`**: Passes a custom function along an axis (Series element-wise or DataFrame row/column-wise).
  ```python
  df["Grade"] = df["Score"].apply(lambda s: "Pass" if s >= 50 else "Fail")
  ```
  > ⚠️ **Best Practice Rule**: Avoid using `.apply()` when Pandas vectorized operations exist (e.g. `df["Sales"] * 2` is 10x-100x faster than `df["Sales"].apply(lambda x: x * 2)`).

---

### 7. Advanced Filtering & `query()`

- **`isin()`**: Filter rows matching a list of allowed values.
  `df[df["Region"].isin(["West", "East"])]`
- **`between()`**: Filter rows falling within an inclusive numerical range.
  `df[df["Sales"].between(500, 1000)]`
- **`query()`**: Expression string syntax for clean conditional queries.
  `df.query("Sales > 500 and Region == 'West'")`

---

### 8. Reshaping with `pivot_table()`
Pivot tables aggregate rectangular data into a matrix representation (Excel Pivot-style):

```python
pivot = pd.pivot_table(
    df,
    values="Sales",
    index="Region",
    columns="Category",
    aggfunc="sum",
    fill_value=0
)
```

---

### 9. Time-Series Handling (`pd.to_datetime()`)
Parse string dates into datetime objects and extract monthly/quarterly periods:
```python
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Month"] = df["Order_Date"].dt.to_period("M")
monthly_revenue = df.groupby("Month")["Revenue"].sum()
```

---

## ❓ 25 Technical Interview Questions & Answers

### Q1: What is `groupby()` in Pandas and how does the Split-Apply-Combine pattern work?
**Answer**: `groupby()` splits a DataFrame into distinct groups based on unique values in specified key column(s), applies an aggregation or transformation function independently to each group, and combines the resulting metrics back into a single output data structure.

### Q2: What is the difference between `.groupby().sum()` and `.groupby().agg(['sum', 'mean'])`?
**Answer**: Direct `.sum()` applies a single aggregation function across all numeric columns. `.agg(['sum', 'mean'])` allows applying multiple aggregation functions simultaneously, returning a DataFrame with hierarchical column headers (or custom named columns if using named aggregations).

### Q3: What is named aggregation in Pandas, and why is it preferred over traditional `.agg()`?
**Answer**: Named aggregation uses keyword syntax inside `.agg()` (e.g. `total_sales=("Sales", "sum")`). It is preferred because it explicitly names output columns and avoids multi-index column headers, resulting in cleaner downstream code.

### Q4: Why is `reset_index()` frequently called after a `groupby()` operation?
**Answer**: `groupby()` moves grouping key columns into the DataFrame's index. Calling `reset_index()` converts these index labels back into standard DataFrame columns with a default integer index (`0, 1, 2...`), making it easier to filter, export, or merge.

### Q5: What is the difference between `pd.merge()` and `df.join()`?
**Answer**: `pd.merge()` joins two DataFrames primarily using column values as join keys (like SQL `JOIN`). `df.join()` joins DataFrames primarily using row indexes as join keys.

### Q6: Explain the difference between `inner`, `left`, `right`, and `outer` joins in `pd.merge()`.
**Answer**:
- `inner`: Returns only rows with matching key values in both DataFrames.
- `left`: Retains all rows from the left DataFrame, filling missing right values with `NaN`.
- `right`: Retains all rows from the right DataFrame, filling missing left values with `NaN`.
- `outer`: Retains all rows from both DataFrames, filling un-matched entries with `NaN`.

### Q7: What is the difference between `pd.concat()` with `axis=0` versus `axis=1`?
**Answer**: `axis=0` stacks DataFrames vertically (adding rows). `axis=1` aligns DataFrames horizontally side-by-side (adding columns matching by index).

### Q8: What does `ignore_index=True` do in `pd.concat()`?
**Answer**: It discards existing row indexes of the input DataFrames and creates a new sequential integer index (`0, 1, 2...`) for the concatenated result.

### Q9: When should you use `map()`, `replace()`, or `apply()` on a Pandas Series?
**Answer**:
- `map()`: Use for 1-to-1 dictionary lookups or single-input Series transformations.
- `replace()`: Use for swapping specific target values across a Series or DataFrame without affecting un-matched values.
- `apply()`: Use when applying custom complex Python functions that cannot be vectorized.

### Q10: Why should vectorized Pandas operations be preferred over `apply()`?
**Answer**: Vectorized operations run in compiled C/NumPy code operating on entire array memory blocks simultaneously. `apply()` executes a Python loop over each element, which incurs high interpreter overhead and is significantly slower.

### Q11: What is `pivot_table()` in Pandas and how does it differ from `groupby()`?
**Answer**: `pivot_table()` reshapes flat tabular data into a 2D grid matrix where unique values of one column become row indexes and unique values of another column become column headers. `groupby()` produces flat stacked multi-index summaries.

### Q12: How do you handle missing values (`NaN`) generated in a `pivot_table()`?
**Answer**: Use the `fill_value` parameter: `pd.pivot_table(..., fill_value=0)`.

### Q13: What does the `df.query()` method do, and what are its advantages?
**Answer**: `df.query()` evaluates conditional filtering expressions using string queries (e.g. `df.query("Sales > 500 and Region == 'West'")`). It eliminates repetitive `df[...]` syntax and enhances code readability.

### Q14: How do you filter rows based on membership in a list of allowed values?
**Answer**: Use the `.isin()` method: `df[df["Category"].isin(["Electronics", "Furniture"])]`.

### Q15: How do you filter values within a specific numerical range in Pandas?
**Answer**: Use the `.between()` method: `df[df["Sales"].between(100, 500, inclusive="both")]`.

### Q16: How do you convert a string column into datetime objects in Pandas?
**Answer**: Use `pd.to_datetime(df["date_str"], format="%Y-%m-%d", errors="coerce")`.

### Q17: How do you extract monthly period periods from a datetime column?
**Answer**: Access the `.dt` accessor: `df["Order_Date"].dt.to_period("M")` or `df["Order_Date"].dt.month`.

### Q18: What is Average Order Value (AOV) and how is it calculated in Pandas?
**Answer**: AOV is total revenue divided by total order count: `df["Revenue"].sum() / df["Order_ID"].nunique()`.

### Q19: How do you find the highest revenue product in each region using Pandas?
**Answer**: Group by `Region` and `Product`, sum revenue, sort by revenue descending, and use `groupby("Region").first()` or `idxmax()`.

### Q20: How do you calculate the percentage contribution of each region to total revenue?
**Answer**: Compute total revenue scalar `total = df["Revenue"].sum()`, then divide regional sum Series by `total` and multiply by 100: `(df.groupby("Region")["Revenue"].sum() / total) * 100`.

### Q21: How do you select the top 5 largest orders by revenue in Pandas?
**Answer**: Use `df.nlargest(5, "Revenue")`.

### Q22: How do you select the bottom 5 smallest orders by revenue in Pandas?
**Answer**: Use `df.nsmallest(5, "Revenue")`.

### Q23: What happens if you perform `pd.merge()` on columns with matching names but different data types?
**Answer**: Pandas raises a `ValueError` or type incompatibility error because key join columns must have compatible dtypes for hash alignment.

### Q24: How can you merge two DataFrames when join key columns have different names in left and right tables?
**Answer**: Use `left_on` and `right_on` parameters: `pd.merge(left_df, right_df, left_on="cust_id", right_on="Customer_ID")`.

### Q25: How do you apply different aggregation functions to different columns in a single `groupby()`?
**Answer**: Pass a dictionary mapping column names to function lists to `.agg()`:
`df.groupby("Region").agg({"Revenue": ["sum", "mean"], "Quantity": "sum", "Discount": "mean"})`.
