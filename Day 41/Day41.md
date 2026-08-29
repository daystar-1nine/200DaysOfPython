# 🐍 Day 41/200 – Masterclass Notes: HTTP & Real-World APIs

🎯 **Goal:** Understand web communication protocols—mastering Client-Server architecture, HTTP Request structure (Methods, URL Anatomy, Headers, Query Parameters, Body), HTTP Response status codes (2xx, 3xx, 4xx, 5xx), JSON serialization vs deserialization (`dumps`, `loads`, `dump`, `load`), the Python `requests` library, resilient API client construction, HTTP mocking in unit tests, and building a professional **REST API Explorer SDK CLI**.

---

## 📌 Executive Summary & Key Takeaways

- **Client-Server Architecture:**
  - **Client (Browser/Python App):** Sends an **HTTP Request** over TCP/IP to a remote server.
  - **Server (Web API):** Processes the request and returns an **HTTP Response** containing status codes, response headers, and JSON/HTML payloads.
- **HTTP Methods Breakdown:**
  - **GET:** Retrieve data without side effects.
  - **POST:** Create new resources on the server.
  - **PUT:** Replace an existing resource entirely.
  - **PATCH:** Update specific attributes of a resource partially.
  - **DELETE:** Remove a resource from the server.
- **HTTP Status Code Categories:**
  - **2xx (Success):** `200 OK`, `201 Created`, `204 No Content`.
  - **3xx (Redirection):** `301 Moved Permanently`, `304 Not Modified`.
  - **4xx (Client Error):** `400 Bad Request`, `401 Unauthorized`, `403 Forbidden`, `404 Not Found`, `409 Conflict`, `422 Unprocessable Content`, `429 Too Many Requests`.
  - **5xx (Server Error):** `500 Internal Server Error`, `502 Bad Gateway`, `503 Service Unavailable`.
- **JSON Serialization:**
  - `json.dumps(obj)` / `json.loads(str)`: Serialize Python dict to JSON string / deserialize JSON string to dict.
  - `json.dump(obj, file)` / `json.load(file)`: Write Python dict directly to file / read JSON directly from file stream.
- **HTTP Unit Test Mocking:** Tests should mock HTTP responses using `unittest.mock.patch('requests.get')` to prevent network dependencies, rate limits, and test flakiness.

---

## 📖 Topic 1: URL Anatomy & HTTP Request Payload

```text
https://api.example.com/users/123?active=true&page=2
  │             │           │               │
Protocol      Host        Path       Query Parameters
```

```python
import requests

url = "https://jsonplaceholder.typicode.com/posts"
headers = {"Authorization": "Bearer sample_token", "Accept": "application/json"}
params = {"userId": 1}
payload = {"title": "Day 41 Python", "body": "HTTP Masterclass", "userId": 1}

# POST Request with Headers, Query Params, and JSON Body
response = requests.post(url, headers=headers, params=params, json=payload, timeout=10)

print(f"Status Code: {response.status_code}") # e.g. 201
print(f"Response JSON: {response.json()}")
```

---

## 📖 Topic 2: Exception Handling & API Error Hierarchy

```python
import requests

class APIError(Exception):
    """Base exception for API Client errors."""
    pass

class APINotFoundError(APIError):
    """Raised when server returns 404 Not Found."""
    pass

def fetch_resource(url: str) -> dict:
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 404:
            raise APINotFoundError(f"Resource at '{url}' was not found.")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout as e:
        raise APIError("Request timed out.") from e
    except requests.exceptions.RequestException as e:
        raise APIError(f"HTTP Request failed: {e}") from e
```

---

## ⚡ Master Cheat Sheet

```python
# HTTP & Requests Cheat Sheet

# 1. GET Request
resp = requests.get("https://api.example.com/users", params={"page": 1})
data = resp.json()

# 2. POST Request
resp = requests.post("https://api.example.com/users", json={"name": "Suraj"})

# 3. PUT vs PATCH
resp_put = requests.put("https://api.example.com/users/1", json={"name": "Suraj", "age": 21}) # Full Replace
resp_patch = requests.patch("https://api.example.com/users/1", json={"age": 22})            # Partial Update

# 4. DELETE Request
resp_del = requests.delete("https://api.example.com/users/1")

# 5. Check Status & Raise Exception
resp.raise_for_status()
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Confusing `params=` and `json=` in `requests`:**
   - ❌ Passing request body to `params=` (e.g. `requests.post(url, params=data)`) attaches attributes to URL string instead of sending JSON payload.
   - ✅ Use `params=` for GET URL query strings and `json=` for POST/PUT/PATCH payload bodies.

2. **Forgetting `timeout=` in HTTP Requests:**
   - ❌ Executing `requests.get(url)` without `timeout` can hang execution indefinitely if the server stops responding.
   - ✅ Always set `timeout=10` or specify explicit connect/read timeouts.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the main difference between PUT and PATCH HTTP methods?
**Answer:** `PUT` replaces the entire target resource with the new request payload. Any fields omitted in the `PUT` payload will be overwritten or set to null/default. `PATCH` applies partial modifications to the resource, updating only the specific fields included in the request payload.

### Q2: Why is HTTP unit test mocking important?
**Answer:** Unit tests should be fast, deterministic, and runnable offline. Real HTTP network calls slow down test execution, depend on external server uptime, risk hitting API rate limits, and can introduce side effects on remote databases.

---

## 📝 Recap Checklist
- [x] Mastered Client-Server HTTP architecture.
- [x] Understood HTTP request methods (GET, POST, PUT, PATCH, DELETE) and status code ranges (2xx, 3xx, 4xx, 5xx).
- [x] Mastered JSON serialization (`dumps`/`loads`) vs stream file operations (`dump`/`load`).
- [x] Built practice task exercises (Tasks 1–9).
- [x] Designed SDK resource client architecture (`client.users.list()`, `client.posts.create()`).
- [x] Created Pytest test suite with HTTP mocks for 15+ test cases.
