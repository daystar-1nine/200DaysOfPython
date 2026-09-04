# 🐍 DAY 59 — Advanced Pandas: Transform, Rank, Rolling & EDA Foundations

## 🏆 Introduction
Welcome to **Day 59**. Yesterday, you mastered data cleaning, handling missing values, deduplication, and string normalization. Today, we step into **Exploratory Data Analysis (EDA)**.

EDA is the systematic process of discovering patterns, detecting anomalies, testing hypotheses, and verifying assumptions using summary statistics and graphical representations.

---

## 🔑 Key Concepts & Technical Reference

### 1. `groupby().transform()` vs `groupby().mean()`
- `df.groupby("Category")["Sales"].mean()` returns a collapsed Series indexed by unique categories (reduces rows).
- `df.groupby("Category")["Sales"].transform("mean")` returns a Series of identical length to the original DataFrame, broadcasting category averages across every row aligned by original index.

```python
# Broadcast group mean across original rows
df["Category_Avg"] = df.groupby("Category")["Sales"].transform("mean")
df["Diff_From_Avg"] = df["Sales"] - df["Category_Avg"]
df["Above_Avg"] = df["Sales"] > df["Category_Avg"]
```

### 2. Intra-Group Ranking with `rank()`
`rank()` computes numerical ranks (1, 2, 3...) for entries in a Series or DataFrame.
- `method="average"`: Default, assigns average rank to tied items.
- `method="min"`: Assigns lowest rank to tied items.
- `method="max"`: Assigns highest rank to tied items.
- `method="first"`: Ranks tied items based on order of appearance.
- `method="dense"`: Like `min`, but rank always increases by 1 for next group (no skipped rank numbers).

```python
# Rank items within category partitions
df["Category_Rank"] = df.groupby("Category")["Sales"].rank(ascending=False, method="dense")
```

### 3. Time-Series Shifts & Differences: `shift()`, `diff()`, `pct_change()`
- `shift(1)`: Shifts data down by 1 row (allows comparison with previous row).
- `diff()`: Equivalent to `df[col] - df[col].shift(1)`.
- `pct_change()`: Computes percentage change: `(current - previous) / previous * 100`.

```python
df["Prev_Sales"] = df["Sales"].shift(1)
df["Sales_Diff"] = df["Sales"].diff()
df["MoM_Growth_%"] = df["Sales"].pct_change() * 100.0
```

### 4. Moving Window Analytics: `rolling()`
`rolling(window=N)` creates a moving window over $N$ consecutive rows for computing smoothed rolling metrics.

```python
df["7_Day_Rolling_Avg"] = df["Sales"].rolling(window=7).mean()
df["3_Month_Rolling_Sum"] = df["Sales"].rolling(window=3).sum()
```

### 5. Cumulative Tracking: `cumsum()` and `cumcount()`
- `cumsum()`: Cumulative running sum along rows.
- `cumcount()`: Returns integer index (0, 1, 2...) of each row within its group.

```python
df["Cum_Sales"] = df["Sales"].cumsum()
df["Customer_Order_Seq"] = df.groupby("Customer_ID").cumcount() + 1
```

### 6. Quantiles & Interquartile Range (IQR) Outlier Detection
Quantiles divide distributions into equal intervals.
- $Q1$ (25th Percentile): 25% of data points fall below this value.
- $Q3$ (75th Percentile): 75% of data points fall below this value.
- $IQR = Q3 - Q1$: Interquartile Range.
- Outlier Bounds:
  $$\text{Lower Bound} = Q1 - 1.5 \times IQR$$
  $$\text{Upper Bound} = Q3 + 1.5 \times IQR$$

```python
Q1 = df["Sales"].quantile(0.25)
Q3 = df["Sales"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["Sales"] < lower_bound) | (df["Sales"] > upper_bound)]
```

### 7. Correlation & Covariance
- `df.corr(numeric_only=True)`: Pearson correlation coefficient $r \in [-1, 1]$ measuring linear dependence between continuous variables.
- `df.cov(numeric_only=True)`: Covariance measuring directional relationship magnitude (scale-dependent).

---

## ❓ 30 Technical Interview Questions & Answers

### Pandas Mechanics (Q1–Q10)
1. **What is the difference between `groupby().agg()` and `groupby().transform()`?**
   - `agg()` aggregates each group into a single summary value per group, producing a reduced output index. `transform()` performs a computation per group and returns a Series aligned with the original input shape.

2. **What does `transform()` return?**
   - A Series or DataFrame with the exact same length and index as the original input dataset.

3. **How does `rank(method="dense")` differ from `rank(method="min")`?**
   - `min` gives tied values the same minimum rank and skips subsequent numbers (e.g. 1, 2, 2, 4). `dense` assigns tied values the same minimum rank but does not skip rank numbers (e.g. 1, 2, 2, 3).

4. **How do you rank rows inside separate group partitions in Pandas?**
   - Use `df.groupby("Group_Col")["Value_Col"].rank(ascending=False)`.

5. **What is the effect of `shift(-1)` versus `shift(1)`?**
   - `shift(1)` moves values down by 1 row (exposing previous row's value). `shift(-1)` moves values up by 1 row (exposing next row's value).

6. **What is the relationship between `diff()` and `shift()`?**
   - `df["col"].diff()` is functionally identical to `df["col"] - df["col"].shift(1)`.

7. **How does `pct_change()` handle initial missing values?**
   - The first element will always be `NaN` because there is no preceding observation to compute percentage growth against.

8. **What does `df["Sales"].rolling(window=7, min_periods=1).mean()` do?**
   - Computes a 7-day moving average, but allows calculation for windows with at least 1 non-null value rather than requiring full 7 values.

9. **What is the difference between `cumsum()` and `cumcount()`?**
   - `cumsum()` returns running total sum of numeric values. `cumcount()` returns 0-indexed integer ordinal position of items within a group.

10. **Why must data be sorted by date before applying `rolling()` or `shift()`?**
    - `rolling()` and `shift()` operate strictly on positional row order. If timestamps are unsorted, moving window metrics and lag differences will be chronologically invalid.

### Statistical & Outlier Concepts (Q11–Q20)
11. **What is a quantile?**
    - A cut point dividing the range of a probability distribution or dataset into continuous intervals with equal probabilities.

12. **What are Q1 and Q3 in descriptive statistics?**
    - Q1 is the 25th percentile (median of lower half). Q3 is the 75th percentile (median of upper half).

13. **How is Interquartile Range (IQR) calculated?**
    - $IQR = Q3 - Q1$.

14. **How are lower and upper outlier bounds defined using IQR?**
    - $\text{Lower} = Q1 - 1.5 \times IQR$, $\text{Upper} = Q3 + 1.5 \times IQR$.

15. **Is every statistical outlier an error that should be deleted?**
    - No. Outliers can represent genuine high-value sales, fraud, viral events, or extreme market shifts. Deleting valid outliers distorts true business variance.

16. **What is the Pearson correlation coefficient?**
    - A normalized statistic between -1 and +1 measuring the strength and direction of linear relationship between two continuous variables.

17. **What is the difference between correlation and covariance?**
    - Covariance measures direction of joint variability but depends on units. Correlation normalizes covariance by standard deviations, making it dimensionless and comparable [-1, 1].

18. **Why does correlation not imply causation?**
    - High correlation can occur due to coincidence or confounding variables (e.g. ice cream sales and sunburns both correlated with summer heat).

19. **What does a standard deviation of 0 indicate for a numerical variable?**
    - Every observation in the dataset has the exact same value (zero variability).

20. **How does median differ from mean in skewed distributions?**
    - Mean is heavily pulled by extreme outliers, whereas median is robust (resistant) to extreme values.

### Exploratory Data Analysis (EDA) Workflow (Q21–Q30)
21. **What is Exploratory Data Analysis (EDA)?**
    - An iterative approach to analyzing data sets to summarize their main characteristics, unveil structural patterns, test hypotheses, and uncover anomalies.

22. **Why conduct EDA prior to machine learning modeling?**
    - EDA reveals data quality flaws, colinearity, missing patterns, non-linear relationships, and outliers that inform feature engineering and model selection.

23. **What is the first step when exploring a new tabular dataset?**
    - Inspecting data dimensions (`df.shape`), column data types (`df.dtypes`), top records (`df.head()`), missing values (`df.isna().sum()`), and duplicated records.

24. **How do you perform categorical EDA in Pandas?**
    - Use `df["col"].value_counts(normalize=True)` to inspect class balances and frequency distributions.

25. **How do you compare categorical subgroups in continuous metrics?**
    - Use `groupby()` with multi-metric `agg()`, box plot distributions, or grouped `transform()` comparisons.

26. **What is the difference between record count and financial volume in categorical EDA?**
    - A category may represent 50% of order counts but only 5% of total revenue. Frequency does not equal financial impact.

27. **How do you analyze seasonality in time-series transactional datasets?**
    - Resample or group by Year/Month/DayOfWeek and compute aggregated totals, percentage growth (`pct_change`), and moving averages (`rolling`).

28. **What makes an EDA report actionable for business stakeholders?**
    - Translating statistical figures into contextual business narrative (e.g. "Category X yields 40% higher margin despite lower order volume").

29. **What is the difference between univariate and bivariate EDA?**
    - Univariate EDA examines distribution of a single column (histograms, mean/std). Bivariate EDA examines relationships between two columns (correlation, cross-tabulation).

30. **How do you handle zero or negative prices during EDA?**
    - Flag them during domain validation, log total count, investigate upstream root causes (discounts vs errors), and filter or correct per business rules.
