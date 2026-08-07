# 🐍 Day 18/200 – Masterclass Notes: JSON, CSV & REST APIs

🎯 **Goal:** Master structured data serialization using **JSON** and **CSV**, and consume web services via **REST APIs** using Python's `json`, `csv`, and `requests` modules.

---

## 📌 Executive Summary & Key Takeaways

- **JSON (JavaScript Object Notation):** A text-based, human-readable data-interchange format. In Python:
  - `json.dumps(obj)`: Serializes Python `dict` $\rightarrow$ JSON `str`.
  - `json.loads(str)`: Deserializes JSON `str` $\rightarrow$ Python `dict`.
  - `json.dump(obj, file)`: Writes Python `dict` $\rightarrow$ JSON file.
  - `json.load(file)`: Reads JSON file $\rightarrow$ Python `dict`.
- **CSV (Comma-Separated Values):** Tabular data storage format.
  - `csv.writer(f)` & `csv.DictWriter(f)`: Writes rows into CSV files (always set `newline=""` on Windows).
  - `csv.reader(f)` & `csv.DictReader(f)`: Reads rows from CSV files.
- **REST APIs (Representational State Transfer):** Web services exposing endpoints for application communication.
  - HTTP GET: Fetches data.
  - HTTP Status Codes: `200` (OK), `201` (Created), `400` (Bad Request), `401` (Unauthorized), `404` (Not Found), `500` (Server Error).
  - Consumption: `requests.get(url, params=params)`.

---

## 📖 Topic 1: JSON Processing (`json` module)

### 1.1 In-Memory Serialization vs File I/O

```python
import json

# Python Dictionary
student_data = {"name": "Suraj", "age": 20, "skills": ["Python", "C++"]}

# 1. Dict -> JSON String (dumps)
json_str = json.dumps(student_data, indent=4)
print(json_str)

# 2. JSON String -> Dict (loads)
parsed_dict = json.loads(json_str)
print(parsed_dict["skills"])  # ['Python', 'C++']

# 3. Dict -> JSON File (dump)
with open("student.json", "w", encoding="utf-8") as f:
    json.dump(student_data, f, indent=4)

# 4. JSON File -> Dict (load)
with open("student.json", "r", encoding="utf-8") as f:
    data_from_file = json.load(f)
```

---

## 📖 Topic 2: CSV Processing (`csv` module)

### 2.1 Writing and Reading Tabular Data

```python
import csv

# Writing CSV using DictWriter
fieldnames = ["Name", "Age", "City"]
students = [
    {"Name": "Suraj", "Age": 20, "City": "Mumbai"},
    {"Name": "Rahul", "Age": 21, "City": "Pune"}
]

# Note: newline="" prevents blank line gaps on Windows
with open("students.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(students)

# Reading CSV using DictReader
with open("students.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"{row['Name']} lives in {row['City']}")
```

---

## 📖 Topic 3: Consuming REST APIs (`requests` library)

### 3.1 Making HTTP GET Requests & Handling Errors

```python
import requests

url = "https://api.github.com/users/daystar-1nine"

try:
    response = requests.get(url, timeout=5)
    # Raises HTTPError if status code is 4xx or 5xx
    response.raise_for_status()

    # Parse JSON payload directly
    user_data = response.json()
    print(f"User: {user_data['name']}, Repos: {user_data['public_repos']}")

except requests.exceptions.RequestException as e:
    print(f"API Request Failed: {e}")
```

---

## ⚡ Master Cheat Sheet

```python
# JSON, CSV & REST API Cheat Sheet

import json, csv, requests

# 1. API GET with Query Parameters
params = {"q": "python", "sort": "stars"}
res = requests.get("https://api.github.com/search/repositories", params=params)
if res.status_code == 200:
    items = res.json()["items"]

# 2. Saving API Data to JSON File
with open("repos.json", "w", encoding="utf-8") as f:
    json.dump(items[:5], f, indent=4)

# 3. Converting JSON API Data to CSV File
with open("repos.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Repo Name", "Stars", "URL"])
    for repo in items[:5]:
        writer.writerow([repo["name"], repo["stargazers_count"], repo["html_url"]])
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Forgetting `newline=""` in CSV Files on Windows:**
   - ❌ `open("data.csv", "w")` (Causes extra blank lines between rows on Windows OS).
   - ✅ Always use `open("data.csv", "w", newline="", encoding="utf-8")`.

2. **Not Handling API Timeout or Network Errors:**
   - ❌ `requests.get(url)` without timeout can block execution indefinitely if remote server hangs.
   - ✅ Always specify `requests.get(url, timeout=5)` and wrap in `try-except requests.exceptions.RequestException`.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the difference between `json.dumps()` and `json.dump()`?
**Answer:** `json.dumps()` (Dump String) serializes a Python object into an in-memory JSON formatted string. `json.dump()` (Dump File) serializes a Python object directly into an open writable file stream.

### Q2: What does `response.raise_for_status()` do in the `requests` library?
**Answer:** It checks `response.status_code`. If the status code indicates an HTTP error (400 to 599), it raises a `requests.exceptions.HTTPError` exception. If the status is 200-299 (OK/Created), it does nothing.

---

## 📝 Recap Checklist
- [x] Serialized Python dicts to JSON strings and files (`dumps`, `dump`).
- [x] Deserialized JSON strings and files to Python dicts (`loads`, `load`).
- [x] Read and written CSV files using `csv.reader` and `csv.DictWriter`.
- [x] Consumed REST APIs using `requests.get()` with query parameters.
- [x] Processed HTTP status codes and handled network exceptions gracefully.
- [x] Built a GitHub Profile Viewer, Weather App, and Currency Converter CLI.
