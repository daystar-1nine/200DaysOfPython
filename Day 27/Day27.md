# 🐍 Day 27/200 – Masterclass Notes: Working with Real-World APIs

🎯 **Goal:** Master professional API interaction in Python—HTTP methods (`GET`, `POST`, `PUT`, `PATCH`, `DELETE`), status codes, headers, query parameters, JSON payloads, `raise_for_status()`, timeouts, retries with exponential backoff, rate limit handling, secure credential management (`.env`), session reuse (`requests.Session`), response caching, and modular API client architecture.

---

## 📌 Executive Summary & Key Takeaways

- **HTTP Method Mapping to CRUD:**
  - `GET` (Read): Retrieve resources without side-effects.
  - `POST` (Create): Submit new data payload to server (`json=data`).
  - `PUT` (Update): Overwrite existing resource entirely.
  - `PATCH` (Partial Update): Modify specific attributes of an existing resource.
  - `DELETE` (Delete): Remove resource.
- **HTTP Status Codes:**
  - `200 OK` / `201 Created` / `204 No Content`: Successful operations.
  - `400 Bad Request` / `401 Unauthorized` / `403 Forbidden` / `404 Not Found`: Client-side errors.
  - `429 Too Many Requests`: Rate limit exceeded.
  - `500 Internal Error` / `502 Bad Gateway` / `503 Service Unavailable`: Server-side errors.
- **Resilience Engineering:**
  - **Timeouts:** ALWAYS pass explicit `timeout=10` to prevent thread hanging indefinitely.
  - **Retries & Exponential Backoff:** Retry transient network errors (e.g. 503, connection timeouts) with doubling delays (`delay = 2 ** attempt`).
  - **Session Reuse (`requests.Session`):** Reuses underlying TCP sockets and persists default headers across multiple API requests.
- **Security & Caching:** Never hardcode Bearer Tokens or API keys in source code. Load via `.env` / `python-dotenv`. Implement local file caching to reduce redundant network round-trips.

---

## 📖 Topic 1: HTTP Client Fundamentals (`requests`)

```python
import requests

# 1. GET with Query Parameters, Custom Headers & Timeout
url = "https://api.github.com/search/repositories"
params = {"q": "python", "sort": "stars", "per_page": 5}
headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "PythonCLIApp"}

try:
    response = requests.get(url, params=params, headers=headers, timeout=10)
    # Raises HTTPError if status_code is 4xx or 5xx
    response.raise_for_status()
    data = response.json()
    print(f"Status Code: {response.status_code} | Total Count: {data['total_count']}")
except requests.Timeout:
    print("Request timed out after 10 seconds.")
except requests.HTTPError as e:
    print(f"HTTP error occurred: {e}")
```

---

## 📖 Topic 2: Reusable API Client Class Architecture

```python
import time
import requests
import logging

class ResilientAPIClient:
    """Production-grade HTTP client with Session reuse, retries, and backoff."""

    def __init__(self, base_url: str, token: str | None = None, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

    def request(self, method: str, endpoint: str, **kwargs) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        kwargs.setdefault("timeout", 10)

        for attempt in range(self.max_retries):
            try:
                resp = self.session.request(method, url, **kwargs)
                resp.raise_for_status()
                return resp.json()
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt == self.max_retries - 1:
                    raise
                backoff = 2 ** attempt
                time.sleep(backoff)
            except requests.HTTPError as e:
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 2))
                    time.sleep(retry_after)
                else:
                    raise
```

---

## ⚡ Master Cheat Sheet

```python
# Real-World API Master Cheat Sheet

import os, requests
from dotenv import load_dotenv

# 1. Environment Credentials
load_dotenv()
token = os.getenv("API_TOKEN")

# 2. Session with Default Authorization Header
session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {token}",
    "Accept": "application/json"
})

# 3. Safe Request Pattern
try:
    resp = session.get("https://api.example.com/data", params={"page": 1}, timeout=5)
    resp.raise_for_status()
    payload = resp.json()
except requests.RequestException as err:
    print(f"API Error: {err}")
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Omitting `timeout` Parameter on Requests:**
   - ❌ `requests.get(url)` (Hangs Python process indefinitely if target server drops connection).
   - ✅ `requests.get(url, timeout=10)`.

2. **Logging Authorization Headers or Secrets:**
   - ❌ `logger.info(f"Sending headers: {headers}")` (Exposes Bearer Tokens in log files).
   - ✅ Strip or mask sensitive keys before logging request headers.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the difference between `PUT` and `PATCH` HTTP methods?
**Answer:** `PUT` replaces the entire target resource with the new payload provided in the request body. `PATCH` applies partial modifications to specific attributes of an existing resource without overwriting unmentioned fields.

### Q2: Why should you use `requests.Session()` instead of raw `requests.get()` calls?
**Answer:** `requests.Session()` maintains an underlying connection pool using HTTP Keep-Alive, significantly reducing latency by avoiding repeated TCP/TLS handshakes across multiple requests, while preserving common headers, parameters, and cookies.

---

## 📝 Recap Checklist
- [x] Utilized HTTP methods (`GET`, `POST`, `PUT`, `PATCH`, `DELETE`).
- [x] Handled status codes and invoked `response.raise_for_status()`.
- [x] Passed query parameters, headers, and JSON payloads.
- [x] Added `timeout=10` to prevent hanging processes.
- [x] Implemented retries with Exponential Backoff (`2 ** attempt`).
- [x] Loaded secrets securely from `.env` via `python-dotenv`.
- [x] Constructed modular `APIClient` with `requests.Session()` and file-based JSON response caching.
- [x] Built complete GitHub CLI application.
