# DAY 58 — PANDAS DATA CLEANING & PREPROCESSING

---

## 🚀 Module Overview

Welcome to Day 58 of the 200 Days of Python Challenge. Today covers one of the most critical skills for Data Scientists and Data Engineers: **Data Cleaning and Preprocessing**.

In real-world business environments, data is rarely clean or ready for modeling. It arrives with missing entries, duplicate rows, whitespace padding, mismatched capitalization, invalid data types, unparseable monetary symbols, impossible ages, and corrupted dates. 

Before performing exploratory data analysis (EDA) or training Machine Learning models, you must build robust, automated data-cleaning pipelines that turn messy raw inputs into pristine, reliable analytical datasets.

---

## 📌 Masterclass Theory Notes

### 1. Understanding Missing Values (`NaN`)
In Pandas, missing numerical or unparsed values are represented by `np.nan` (**Not a Number**).

#### A. Detecting Missing Values
- `df.isna()` or `df.isnull()`: Returns a boolean DataFrame of identical shape (`True` for missing values).
- `df.isna().sum()`: Returns a Series with missing value count per column.
- `(df.isna().mean() * 100).round(2)`: Computes the percentage of missing values per column.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "Name": ["Rahul", "Priya", "Aman"],
    "Age": [20, np.nan, 21]
})

# Missing counts and percentages
print(df.isna().sum())
print(df.isna().mean() * 100)
```

---

### 2. Handling Missing Values (`dropna()` vs `fillna()`)

#### A. Deleting Missing Data (`dropna()`)
- `df.dropna()`: Drops any row containing at least one missing value.
- `df.dropna(subset=["Age"])`: Drops rows where `Age` specifically is missing.
> ⚠️ **Warning**: Never use `.dropna()` blindly on datasets with heavy missingness, as it can destroy substantial valuable data.

#### B. Imputing Missing Values (`fillna()`)
- **Constant**: `df["City"].fillna("Unknown")`
- **Mean Imputation**: `df["Age"].fillna(df["Age"].mean())` (Best for symmetric numeric data).
- **Median Imputation**: `df["Age"].fillna(df["Age"].median())` (Best for skewed numeric data with outliers).
- **Mode Imputation**: `df["City"].fillna(df["City"].mode()[0])` (Best for categorical text columns).
- **Forward Fill (`ffill()`)**: Propagates previous valid value forward (ideal for sequential/time-series data).
- **Backward Fill (`bfill()`)**: Propagates next valid value backward.

---

### 3. Duplicate Detection & Removal

- **Detect Duplicates**: `df.duplicated()` returns a boolean Series (`True` for repeated rows).
- **Count Duplicates**: `df.duplicated().sum()`.
- **Remove Duplicates**: `df.drop_duplicates(keep="first")`.
- **Deduplicate on Specific Keys**:
  ```python
  df.drop_duplicates(subset=["Customer_ID"], keep="first")
  ```

---

### 4. String Cleaning & Categorical Normalization

Vectorized `.str` string accessors allow clean element-wise text manipulation:

| Function | Operation | Example Input | Output |
|---|---|---|---|
| `.str.strip()` | Removes leading & trailing whitespace | `" Rahul "` | `"Rahul"` |
| `.str.lower()` | Converts string to lowercase | `"MUMBAI"` | `"mumbai"` |
| `.str.upper()` | Converts string to uppercase | `"mumbai"` | `"MUMBAI"` |
| `.str.title()` | Capitalizes first letter of each word | `"rahul sawant"` | `"Rahul Sawant"` |
| `.str.replace()` | Replaces target substring | `"₹50,000"` | `"50000"` |
| `.str.contains()` | Searches string patterns | `"Rahul"` | `True` |

```python
# Normalizing text columns
df["City"] = df["City"].astype(str).str.strip().str.title()
```

---

### 5. Cleaning Monetary Data & Numeric Conversion

Monetary data often contains currency symbols (`₹`, `$`) and thousands separators (`,`). Convert them safely using string replacements followed by `pd.to_numeric(errors="coerce")`:

```python
# Clean currency string to float
df["Salary"] = (
    df["Salary"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
```

`errors="coerce"` replaces unparseable strings (e.g. `"unknown"`) with `NaN`, preventing crashes.

---

### 6. Date Parsing & Extracting Components

Use `pd.to_datetime(errors="coerce")` to convert string dates to Pandas Timestamps, then extract components via `.dt`:

```python
df["Join_Date"] = pd.to_datetime(df["Join_Date"], errors="coerce")

# Component extraction
df["Join_Year"] = df["Join_Date"].dt.year
df["Join_Month"] = df["Join_Date"].dt.month
df["Month_Name"] = df["Join_Date"].dt.month_name()
df["Day_Name"] = df["Join_Date"].dt.day_name()
```

---

### 7. Domain Validation & Outlier Filtering

Numeric columns may contain logically impossible values (e.g. Age = -5 or Age = 200). Filter or invalidate out-of-bounds records using `.between()` or boolean masks:

```python
# Setting impossible ages to NaN
df.loc[~df["Age"].between(0, 120), "Age"] = np.nan
```

---

### 8. Professional Data Cleaning Pipeline Architecture

A robust cleaning pipeline executes operations in a deterministic sequence:

```text
RAW DATA INGESTION
       ↓
1. Standardize Column Headers (lowercase, strip, underscores)
       ↓
2. Clean String Columns (trim whitespace, title case)
       ↓
3. Standardize Categorical Values (e.g. M/Male -> Male)
       ↓
4. Coerce Data Types (to_numeric, to_datetime)
       ↓
5. Validate Domain Ranges (Age: 0-120, Salary >= 0)
       ↓
6. Deduplicate Records (drop_duplicates on primary key)
       ↓
7. Impute Missing Values (median for numeric, mode for text)
       ↓
8. Extract Derived Features (join_year, join_month)
       ↓
PROCESSED CLEAN DATA
```

---

## ❓ 30 Technical Interview Questions & Answers

### Q1: What is Data Cleaning and why is it necessary in Data Science?
**Answer**: Data Cleaning is the process of detecting, correcting, or removing corrupt, inaccurate, incomplete, or incorrectly formatted records from a dataset. It is necessary because Machine Learning models and statistical analyses are only as good as the underlying data ("Garbage In, Garbage Out").

### Q2: What does `NaN` represent in Pandas and what is its underlying Python/NumPy type?
**Answer**: `NaN` stands for "Not a Number". In Pandas and NumPy, missing numerical data is represented by `float('nan')` (or `np.nan`), which has a floating-point data type (`float`).

### Q3: What is the difference between `.isna()` and `.isnull()` in Pandas?
**Answer**: They are exact aliases of each other. `.isnull()` was inherited from R syntax compatibility, while `.isna()` aligns with NumPy's `isnan()`.

### Q4: How do you calculate the percentage of missing values per column in a DataFrame?
**Answer**: Call `(df.isna().mean() * 100).round(2)` or `(df.isna().sum() / len(df) * 100).round(2)`.

### Q5: What is the risk of using `df.dropna()` without column scoping?
**Answer**: Unscoped `df.dropna()` removes any row that contains even a single missing value across any column. If missing values are scattered across many columns, this can discard a huge portion of the dataset unnecessarily.

### Q6: What does `df.dropna(subset=['col1', 'col2'])` do?
**Answer**: It drops rows only if missing values are present specifically in `col1` or `col2`, ignoring NaNs in other columns.

### Q7: Explain Mean vs Median imputation for missing numeric data. When should you choose Median?
**Answer**: Mean imputation replaces missing values with the arithmetic average. Median imputation replaces missing values with the middle value of the sorted distribution. Median imputation is preferred when the numerical variable contains outliers or a skewed distribution, as the mean is heavily pulled by extreme values.

### Q8: What is Mode imputation and when is it applied?
**Answer**: Mode imputation replaces missing entries with the most frequently occurring value in a column. It is primarily applied to categorical (text) or discrete integer columns.

### Q9: How do `ffill()` and `bfill()` work, and for what type of data are they suited?
**Answer**: `ffill()` (forward fill) propagates the last valid value forward. `bfill()` (backward fill) propagates the next valid value backward. They are suited for ordered sequential or time-series data (e.g. stock prices or sensor readings).

### Q10: How do you detect and count duplicate rows in a DataFrame?
**Answer**: Use `df.duplicated()` to obtain a boolean Series of duplicate indicators, and `df.duplicated().sum()` to count the total number of duplicate rows.

### Q11: What is the difference between `keep='first'` and `keep='last'` in `drop_duplicates()`?
**Answer**: `keep='first'` retains the first occurrence of a duplicate row and drops subsequent repetitions. `keep='last'` retains the final occurrence and drops earlier repetitions.

### Q12: How do you strip leading and trailing whitespace from all string columns in a DataFrame?
**Answer**: Iterate through string columns and call `.str.strip()`:
```python
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].astype(str).str.strip()
```

### Q13: What is the difference between `.astype(int)` and `pd.to_numeric(..., errors='coerce')`?
**Answer**: `.astype(int)` raises a `ValueError` if the Series contains unparseable text (e.g. `"unknown"`) or `NaN`. `pd.to_numeric(..., errors='coerce')` gracefully converts unparseable text and invalid strings to `NaN` without crashing.

### Q14: How do you clean monetary strings containing currency symbols like `₹50,000` or `$100` into float numbers?
**Answer**: Remove currency symbols and commas via `.str.replace()`, then convert using `pd.to_numeric()`:
```python
df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace("₹", "").str.replace("$", "").str.replace(",", "").str.strip(), errors="coerce")
```

### Q15: How do you convert a column of date strings to Pandas Timestamps while handling invalid date entries?
**Answer**: Use `pd.to_datetime(df["date_col"], errors="coerce")`. Invalid date strings are converted to `NaT` (Not a Time).

### Q16: How do you extract the month name and day name from a datetime column?
**Answer**: Use the `.dt` accessor: `df["date_col"].dt.month_name()` and `df["date_col"].dt.day_name()`.

### Q17: How do you filter numeric values falling outside a valid domain boundary (e.g. Age outside 0-120)?
**Answer**: Use `.between()`:
`df_clean = df[df["Age"].between(0, 120)]` or invalidate out-of-range values:
`df.loc[~df["Age"].between(0, 120), "Age"] = np.nan`.

### Q18: How do you standardize inconsistent categorical text values like `M`, `male`, `MALE` to `Male`?
**Answer**: Normalize casing with `.str.strip().str.lower()`, then use `.replace()` with a dictionary mapping:
```python
df["Gender"] = df["Gender"].astype(str).str.strip().str.lower().replace({"m": "Male", "male": "Male", "f": "Female", "female": "Female"})
```

### Q19: What does `na=False` do in `df['Name'].str.contains('keyword', na=False)`?
**Answer**: It forces missing (`NaN`) entries in the string column to evaluate to `False` instead of propagating `NaN`, preventing errors during boolean indexing.

### Q20: How do you standardize column header names to lowercase with underscores?
**Answer**:
```python
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
```

### Q21: What is a Data Quality Report and what key metrics should it include?
**Answer**: A Data Quality Report documents the health of a dataset before and after cleaning. It includes total row counts, missing value counts per column, duplicate row counts, invalid value counts, data types, and executed cleaning operations.

### Q22: Why should raw data be preserved separately from processed clean data?
**Answer**: Preserving raw data maintains immutability and auditability. It ensures that cleaning pipelines can be re-executed or updated without permanent loss of original source data.

### Q23: What problem occurs if you perform data imputation before splitting dataset into Train/Test sets in ML?
**Answer**: Imputing before splitting causes **Data Leakage**, where statistical properties (like test set mean/median) leak into the training set, causing overly optimistic model evaluation metrics.

### Q24: How would you clean a messy Phone column containing formats like `98765-43210`, `+91 9876543210`, and `9876543210`?
**Answer**: Strip non-digit characters using regex `.str.replace(r'\D', '', regex=True)` and slice the last 10 digits.

### Q25: What is the difference between `dropna(how='any')` and `dropna(how='all')`?
**Answer**: `how='any'` (default) drops a row if ANY column is missing. `how='all'` drops a row only if ALL columns in that row are missing.

### Q26: What does `df.select_dtypes(include=['number'])` do?
**Answer**: It returns a DataFrame slice containing only numerical (int, float) columns.

### Q27: How do you replace empty string values `""` or whitespace-only strings with `NaN`?
**Answer**: Use `.replace(r'^\s*$', np.nan, regex=True)`.

### Q28: How do you check if a DataFrame has any duplicate primary key entries?
**Answer**: Check `df["Customer_ID"].duplicated().any()`.

### Q29: Why is domain knowledge essential during data cleaning?
**Answer**: Domain knowledge determines sensible business rules (e.g. whether a negative salary is a refund or an error, or whether Age = 0 indicates an infant or missing data).

### Q30: What is the difference between `isna()` and `notna()` in Pandas?
**Answer**: `isna()` returns `True` for missing values. `notna()` returns `True` for valid non-missing values (useful for filtering complete cases: `df[df["Email"].notna()]`).
