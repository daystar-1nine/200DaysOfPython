# Day 53 — Real-World Data Processing, Cleaning, Transformation & Analysis in Python

---

## 📚 Masterclass Overview

Data processing is the process of taking raw, unorganized, or messy datasets and transforming them into structured, validated, cleaned, and actionable formats suitable for analytical insights and reporting. In real-world software engineering, backend applications, data science, and data engineering, raw inputs are rarely clean. Data coming from web forms, legacy databases, CSV exports, or third-party APIs often contain:
- Leading/trailing whitespace or inconsistent casing (`" rahul "`, `"RAHUL"`).
- Missing, empty, or null values (`None`, `""`, `"N/A"`, `"?"`).
- Invalid data types or malformed numbers (`"fifty"`, `"-500"` for price).
- Duplicate transactions or record collisions.
- Unformatted date strings (`"2026/09/01"`, `"01-09-2026"`).

Today's masterclass establishes the foundational data pipeline mindset required before moving into large-scale libraries like **NumPy** and **Pandas**.

---

## 🧠 Core Concepts & Technical Architecture

### 1. The Data Processing Pipeline
A clean software design pattern separates data operations into modular, single-responsibility stages:

```text
Raw Dataset (CSV/JSON/API)
           │
           ↓
     Data Inspection
           │
           ↓
     Data Cleaning (String Normalization, Safe Parsing)
           │
           ↓
     Data Validation (Domain Boundary & Schema Rules)
           │
           ↓
     Deduplication & Transformation (Derived Fields)
           │
           ↓
     Statistical Aggregation & Analysis
           │
           ↓
     Reporting & Clean Export (CSV/Text Report)
```

---

### 2. String Cleaning & Normalization
Uncleaned strings introduce subtle bugs during equality checks, grouping, and deduplication.
- `str.strip()` removes unwanted leading and trailing whitespace.
- `str.title()`, `str.lower()`, `str.upper()` enforce consistent casing.

```python
raw_name = "   rahul sawant   "
clean_name = raw_name.strip().title()  # Output: "Rahul Sawant"
```

---

### 3. Safe Type Conversion Patterns
Directly executing `int(value)` or `float(value)` risks raising unhandled `ValueError` crashes when processing untrusted inputs. Safe converters wrap conversions defensively:

```python
def safe_int(value: str | None, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return default

def safe_float(value: str | None, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return default
```

---

### 4. Handling Missing & Malformed Values
Strategies for missing data depending on domain requirements:
1. **Drop Record**: Exclude invalid or missing critical fields (e.g. missing primary key or price).
2. **Default Imputation**: Replace missing values with standard defaults (`"Unknown"`, `0.0`).
3. **Statistical Imputation**: Fill missing numeric values with column mean/median.

---

### 5. Deduplication Techniques
Record deduplication ensures that duplicate orders or log entries do not distort financial aggregations.
- **In-Memory Unique Tracking**:
```python
seen_ids = set()
unique_records = []
for record in records:
    if record.order_id not in seen_ids:
        seen_ids.add(record.order_id)
        unique_records.append(record)
```

---

### 6. Derived Properties & Data Transformation
Derived properties compute new values from raw attributes.
Using Python `@dataclass` properties allows on-the-fly calculation without duplicating state:

```python
from dataclasses import dataclass
from datetime import date

@dataclass
class Sale:
    order_id: int
    customer: str
    product: str
    category: str
    price: float
    quantity: int
    date: date

    @property
    def total(self) -> float:
        """Calculates derived order total amount."""
        return self.price * self.quantity
```

---

### 7. Data Aggregation & Grouping
Efficient analytical operations leverage standard library structures:
- `min()` / `max()` with key functions for extremal record identification:
  ```python
  highest_sale = max(sales, key=lambda s: s.total)
  ```
- `collections.defaultdict` for category & product revenue grouping:
  ```python
  from collections import defaultdict
  category_totals = defaultdict(float)
  for s in sales:
      category_totals[s.category] += s.total
  ```
- `collections.Counter` for frequency metrics:
  ```python
  from collections import Counter
  product_counts = Counter(s.product for s in sales)
  ```

---

### 8. Python In-Memory JOINs (Data Engineering Basics)
When working with multiple relational datasets without a SQL database, Python dictionaries act as hash maps for $O(1)$ key lookups to perform inner/left joins:

```python
# Join sales.csv with customers.csv on customer_name or customer_id
customer_lookup = {c["id"]: c for c in customers}
joined_sales = [
    {**sale, "city": customer_lookup[sale["customer_id"]]["city"]}
    for sale in sales if sale["customer_id"] in customer_lookup
]
```

---

## ❓ 20 Technical Interview Questions & Answers

### Q1: What is the primary difference between raw data and clean data?
**Answer:** Raw data refers to unedited, unstructured, or original data as ingested directly from sources (APIs, logs, user input, CSVs). It often contains errors, whitespace, missing values, duplicates, and inconsistent types. Clean data has undergone inspection, normalization, validation, deduplication, and type casting, rendering it trustworthy and ready for analysis.

### Q2: Why is separating data pipeline stages (Inspect, Clean, Validate, Transform, Analyze) considered best practice?
**Answer:** Following the Single Responsibility Principle prevents monolithic functions that attempt to perform IO, validation, modification, and presentation simultaneously. Modular pipeline stages make the system testable, maintainable, debuggable, and reusable across different datasets.

### Q3: What is a safe type conversion function and why is it preferred over direct casting?
**Answer:** Direct type casting like `float("invalid")` raises an unhandled `ValueError` that terminates the program. A safe conversion function defensively catches parsing exceptions and returns `None` or a specified default value, allowing the pipeline to flag or clean invalid values gracefully without crashing.

### Q4: How do leading and trailing whitespaces impact data analysis if left uncleaned?
**Answer:** Extra whitespaces cause string comparison failures (`"Laptop"` != `"Laptop "`), corrupting grouping operations (`defaultdict`, SQL `GROUP BY`), breaking deduplication sets, and causing unexpected user interface display issues.

### Q5: Name four common strategies for handling missing values in a dataset.
**Answer:**
1. **Record Removal (Deletion)**: Exclude records missing essential fields.
2. **Constant Imputation**: Replace missing values with defaults (e.g., `"N/A"`, `0`).
3. **Statistical Imputation**: Fill missing numeric values with the mean, median, or mode.
4. **Predictive/Forward Fill**: Fill missing values using preceding values in time-series data.

### Q6: How do you identify duplicate records in Python without external libraries?
**Answer:** By maintaining a `set()` of seen unique identifiers (e.g., `order_id`) while iterating through records, or by overriding `__hash__` and `__eq__` methods on dataclasses to check full object equality.

### Q7: What is a derived column or derived property in data processing?
**Answer:** A derived column is a new value calculated dynamically from pre-existing raw attributes rather than stored statically. For example, `total = price * quantity` or `age = current_year - birth_year`.

### Q8: How does Python's `sorted()` function sort objects by multiple conditions?
**Answer:** By supplying a tuple in the `key` lambda function: `sorted(students, key=lambda s: (s.marks, s.name), reverse=True)`. Python compares tuple items element-by-element, sorting primarily by marks and breaking ties with the name attribute.

### Q9: When should you use `collections.defaultdict(float)` over a standard dictionary for aggregations?
**Answer:** A `defaultdict(float)` automatically initializes missing keys with `0.0` upon first access. This eliminates explicit key presence checks (`if key not in d: d[key] = 0.0`), resulting in cleaner, faster aggregation loops.

### Q10: How does `collections.Counter` simplify frequency analysis?
**Answer:** `Counter` is a subclass of `dict` designed for counting hashable objects. Passing an iterable auto-populates element frequencies, and `.most_common(n)` retrieves the top $N$ frequent elements efficiently.

### Q11: What is the computational complexity of `max()` with a key function versus sorting a list to find the top item?
**Answer:** `max()` runs in $O(N)$ linear time by maintaining a single top variable in a single pass. Sorting the list using `sorted()` takes $O(N \log N)$ time, making `max()` significantly more efficient when only the single extremal element is needed.

### Q12: Why should dataclasses be preferred over plain dictionaries for raw data transformation?
**Answer:** Dataclasses provide static type hints, auto-generated constructors, clear attribute documentation, IDE autocompletion, immutability options (`frozen=True`), and encapsulation of computed `@property` methods.

### Q13: What is the risk of modifying a raw dataset file directly in place during cleaning?
**Answer:** In-place modification destroys original raw data history, preventing auditability, re-processing with updated business logic, or recovering from buggy cleaning rules. Raw data should be treated as immutable read-only inputs, writing outputs to separate processed files.

### Q14: How can you perform an inner join between two CSV datasets in pure Python?
**Answer:** Load the right dataset into a dictionary indexed by join key (e.g., `{customer_id: customer_dict}`). Iterate over the left dataset (sales), looking up matching keys in $O(1)$ time and merging dictionary values.

### Q15: What is the difference between data cleaning and data transformation?
**Answer:** Data cleaning fixes errors, removes noise, standardizes formats, handles missing values, and deletes duplicates. Data transformation restructures clean data into new representations, computes derived metrics, rescales numbers, or aggregates records for analysis.

### Q16: How do you handle date string parsing safely when dates come in multiple formats?
**Answer:** Iterate through a list of acceptable `datetime.strptime` format patterns (e.g., `"%Y-%m-%d"`, `"%d-%m-%Y"`, `"%Y/%m/%d"`) inside a `try...except` block, returning a valid `datetime.date` object on the first matching pattern or raising a validation error if all fail.

### Q17: What is memory-bounded stream processing and when is it required over loading full datasets into memory?
**Answer:** Stream processing reads and processes data line-by-line using file generators or iterators (`for line in file:`), keeping memory overhead minimal ($O(1)$) regardless of file size. It is required when processing datasets larger than available RAM (e.g., 10 GB+ CSVs).

### Q18: What is data normalization in string processing?
**Answer:** Data normalization transforms variable representations of equivalent values into a single uniform format. Examples include converting phone numbers to E.164 standard, lowercasing emails, or mapping categories (`"elect."` -> `"Electronics"`).

### Q19: Why is `lambda` used in `min()`, `max()`, and `sorted()`?
**Answer:** `lambda` creates an anonymous inline function that extracts a specific comparison key from complex objects or dictionaries (e.g., `key=lambda sale: sale.total`), avoiding the need to write separate named functions.

### Q20: How will Pandas build upon the raw Python data processing concepts learned today?
**Answer:** Pandas automates these manual loops by providing vectorised `DataFrame` data structures, built-in methods (`df.dropna()`, `df.fillna()`, `df.drop_duplicates()`, `df.groupby()`, `df.merge()`), and C-optimized C/NumPy execution engines for handling millions of rows effortlessly.
