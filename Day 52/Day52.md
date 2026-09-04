# 🐍 Day 52 — JSON, CSV & Data Serialization

> **"Data serialization is the bridge connecting Python applications to external APIs, databases, message queues, and Data Science pipelines. Mastering JSON and CSV processing enables clean data exchange, transformation, and validation."**

---

## 🗺️ Day 52 Architecture & Roadmap

```text
                                    DAY 52
                                       │
          ┌────────────────────────────┴────────────────────────────┐
          ↓                                                         ↓
        JSON                                                       CSV
          │                                                         │
          ↓                                                         ↓
  Serialization & Deserialization                         Reading & Writing
  (dump/dumps, load/loads, nested JSON)                   (reader/writer, DictReader/DictWriter)
          │                                                         │
          └────────────────────────────┬────────────────────────────┘
                                       ↓
                           DATACLASS SERIALIZATION
                           (asdict() & **dictionary unpacking)
                                       ↓
                           DATA VALIDATION PIPELINE
                           (Defensive checks & Pydantic connection)
                                       ↓
                     STUDENT DATA MANAGEMENT SYSTEM V2
                     (app/, data/, output/, tests/)
```

---

## 📚 PART 1 — Core Technical Concepts

### 1. Data Serialization vs Deserialization
- **Serialization**: Translating in-memory Python objects (e.g. `dict`, `list`, `@dataclass`) into a standardized byte or string format (e.g., JSON string, CSV text file) for network transmission or disk storage.
- **Deserialization**: Reconstructing standardized strings/files (JSON string, CSV text) back into native, typed Python data structures in memory.

```text
Python Object ───(Serialization)───> JSON/CSV String/File
Python Object <──(Deserialization)── JSON/CSV String/File
```

### 2. Python ↔ JSON Type Mapping
| Python Type | JSON Equivalent Type |
| :--- | :--- |
| `dict` | `object` |
| `list`, `tuple` | `array` |
| `str` | `string` |
| `int`, `float` | `number` |
| `True` / `False` | `true` / `false` |
| `None` | `null` |

> **[!NOTE]**
> `True`, `False`, and `None` in Python convert strictly to lowercase `true`, `false`, and `null` in JSON output strings.

### 3. Built-in `json` Module Functions
- `json.dumps(obj)`: Converts Python object $\rightarrow$ JSON string (`dumps` = **dump string**).
- `json.dump(obj, file)`: Writes Python object directly to an open file stream.
- `json.loads(json_str)`: Parses JSON string $\rightarrow$ Python object (`loads` = **load string**).
- `json.load(file)`: Reads and parses JSON directly from an open file stream.
- **Pretty Formatting**: `json.dumps(data, indent=4)` formats multi-line JSON.

### 4. CSV Operations (`csv` Module)
- **`csv.reader(file)`**: Iterates over CSV rows as lists of strings (`['name', '21', '85.0']`).
- **`csv.writer(file)`**: Writes rows using `.writerow()` or `.writerows()`.
- **`csv.DictReader(file)`**: Parses rows as dictionaries keyed by header column names (`{'name': 'Rahul', 'age': '21'}`).
- **`csv.DictWriter(file, fieldnames=...)`**: Writes dictionaries to CSV after writing headers via `.writeheader()`.

> **[!IMPORTANT]**
> CSV files store all cell data as text strings. You must explicitly convert numeric fields (e.g. `int(row['age'])`, `float(row['marks'])`) upon reading.

### 5. Dataclass Serialization Protocols
- **To Dictionary**: `from dataclasses import asdict; data_dict = asdict(student_instance)`
- **To JSON String**: `json_str = json.dumps(asdict(student_instance))`
- **From Dictionary**: `student_instance = Student(**data_dict)` (kwargs unpacking)

### 6. JSON vs CSV Comparison
| Feature | JSON | CSV |
| :--- | :---: | :---: |
| **Nested / Hierarchical Data** | ✅ Supported | ❌ Flat Tabular Only |
| **Human Readable** | ✅ Excellent | ✅ Excellent |
| **API Payload Standard** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Data Science / Tabular** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Native Data Types** | Numbers, Booleans, Null | Strings Only |

---

## 🎤 PART 2 — Technical Interview Questions & Answers

### Q1: What is JSON, and why is it the dominant data format for Web APIs?
**Answer**: JSON (JavaScript Object Notation) is a lightweight, language-agnostic text format. It is dominant for Web APIs because it natively supports hierarchical nested structures, is easily human-readable, and maps seamlessly to data structures across modern programming languages.

### Q2: What is the difference between `json.dump()` and `json.dumps()`?
**Answer**: `json.dumps()` serializes a Python object into a JSON-formatted `str` in memory. `json.dump()` serializes a Python object directly into a writable file stream.

### Q3: What is the difference between `json.load()` and `json.loads()`?
**Answer**: `json.loads()` deserializes a JSON `str` into a Python object in memory. `json.load()` reads and deserializes JSON content directly from a readable file stream.

### Q4: Why are CSV values initially read as strings in Python?
**Answer**: The CSV file format is plain text without embedded schema or type metadata. Therefore, Python's `csv` module parses all cell values as raw string objects. Type casting (e.g., `int()`, `float()`) must be applied explicitly.

### Q5: What is the difference between `csv.reader` and `csv.DictReader`?
**Answer**: `csv.reader` returns each line as a `list` of column strings indexed by position (e.g. `row[0]`). `csv.DictReader` uses the first row as dictionary keys and returns each subsequent row as a `dict` mapping header names to row values (e.g. `row['name']`).

### Q6: How do you convert a `@dataclass` instance to a JSON string?
**Answer**: Use `dataclasses.asdict()` to convert the dataclass instance into a standard Python dictionary, then pass that dictionary to `json.dumps()`:
```python
from dataclasses import asdict
import json

json_string = json.dumps(asdict(student_obj))
```

### Q7: How do you construct a `@dataclass` instance from a dictionary?
**Answer**: Use double-asterisk dictionary unpacking (`**kwargs`):
```python
student_obj = Student(**data_dict)
```

### Q8: What exception is raised when parsing malformed JSON string data?
**Answer**: Python's `json` module raises a `json.JSONDecodeError` exception if the input text violates JSON syntax rules.

### Q9: Why is external data validation critical prior to deserialization into domain objects?
**Answer**: External inputs (from API payloads, user uploads, or files) can contain missing fields, unexpected types, or out-of-bound values (e.g., negative age or marks > 100). Validation ensures corrupt data is rejected before damaging internal state or database tables.

### Q10: How do today's JSON serialization and validation concepts connect to FastAPI?
**Answer**: FastAPI automatically deserializes incoming JSON HTTP request bodies into Python dictionaries, passes them to Pydantic models for validation, and serializes return Pydantic/dataclass models back into JSON responses.

---

## 📊 PART 3 — Day 52 Checklist & Self-Assessment

- [x] JSON Serialization & Deserialization (`dumps`, `dump`, `loads`, `load`)
- [x] Defensive `json.JSONDecodeError` handling
- [x] Nested JSON structure traversal & Practice 1
- [x] CSV Reading & Writing (`csv.reader`, `csv.writer`, `DictReader`, `DictWriter`)
- [x] Dataclass `asdict()` and `**dict` instantiation
- [x] Data validation pipeline rules
- [x] 5 Standalone Coding Challenges
- [x] Student Data Management System V2 CLI
- [x] 20+ Pytest Automated Tests
