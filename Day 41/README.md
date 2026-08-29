# REST API Explorer

A professional, type-safe Python CLI Application and SDK for exploring and interacting with REST APIs. Built with Python 3.10+ and the `requests` library.

---

## Overview

`REST API Explorer` acts as both an interactive command-line application (TUI/CLI) and a reusable SDK (`client.users.list()`, `client.posts.create()`) for communicating with RESTful web services over HTTP protocol.

---

## Features

- 🌐 **Full HTTP CRUD Operations:** Supports `GET`, `POST`, `PUT`, `PATCH`, and `DELETE` requests.
- 🛠️ **Bonus SDK Architecture:** Includes intuitive sub-resource managers (`client.users`, `client.posts`) modeling modern REST SDK design.
- 📦 **Typed Domain Models:** `User` and `Post` `@dataclass` response models with custom Dunder methods (`__str__`, `__repr__`, `__getitem__`).
- ⏱️ **Performance Profiling:** Includes `@timer` decorator timing HTTP request execution duration.
- 🛡️ **Robust Error Handling:** Custom API exception taxonomy (`APIError`, `APIConnectionError`, `APINotFoundError`, `APITimeoutError`, `APIValidationError`).
- 🧪 **Offline Test Suite:** Complete Pytest test suite using `unittest.mock` to test network client operations fast without live internet connection dependencies.

---

## Technologies

- **Language:** Python 3.10+
- **HTTP Client:** Requests
- **Testing & Mocking:** Pytest, Unittest.mock
- **Architecture:** OOP, Layered Services, Dataclasses, Exception Hierarchy, Decorators

---

## Project Structure

```text
Day 41/
├── tasks/                      # Exercises 1-9 Scripts
│   ├── task1_first_request.py
│   ├── task2_extract_data.py
│   ├── task3_query_params.py
│   ├── task4_post_request.py
│   ├── task5_put_request.py
│   ├── task6_patch_request.py
│   ├── task7_delete_request.py
│   ├── task8_error_handling.py
│   └── task9_api_client_class.py
├── src/
│   └── api_explorer/
│       ├── __init__.py
│       ├── main.py             # CLI Driver
│       ├── client.py           # APIClient & SDK Resources (client.users, client.posts)
│       ├── models.py           # User & Post Dataclasses (with Dunder Methods)
│       ├── services.py         # UserService & PostService
│       ├── exceptions.py       # API Custom Exception Hierarchy
│       └── cli.py              # Interactive Terminal Menu
├── tests/
│   ├── test_models.py
│   ├── test_client.py          # HTTP Mocking Tests
│   └── test_services.py
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Installation

```bash
# Navigate to Day 41 project directory
cd Day\ 41

# Install dependencies in editable mode
pip install -e .
```

---

## API Used

By default, the application connects to the free public test API:
- **JSONPlaceholder API:** `https://jsonplaceholder.typicode.com`

---

## How to Run

Launch the interactive CLI menu:

```bash
python src/api_explorer/main.py
```

Or via package runner:

```bash
python -m api_explorer
```

---

## Example Usage

### SDK Usage
```python
from api_explorer.client import APIClient
from api_explorer.services import UserService

client = APIClient("https://jsonplaceholder.typicode.com")

# SDK Resource syntax
raw_users = client.users.list()
created_post = client.posts.create("New Title", "Body text", user_id=1)

# Service layer syntax
user_service = UserService(client)
user = user_service.get_user(1)
print(user)
```

### CLI Terminal Output
```text
╔══════════════════════════════════╗
║       REST API EXPLORER          ║
╚══════════════════════════════════╝
1. List Users
2. Get User
3. Create Post
4. Update Post
5. Delete Post
6. Search Users
7. Exit

Select Option (1-7): 1

Users
──────────────────────────────────────────────────
ID     NAME                      EMAIL
──────────────────────────────────────────────────
1      Leanne Graham             Sincere@april.biz
2      Ervin Howell              Shanna@melissa.tv
3      Clementine Bauch          Nathan@yesenia.net
```

---

## Error Handling

All HTTP network failures and status codes are trapped and mapped to clean domain exceptions:

```python
try:
    user = client.users.get(9999)
except APINotFoundError as e:
    print(f"User not found: {e}")
except APIConnectionError as e:
    print(f"Network error: {e}")
```

---

## Testing

Run unit tests (all HTTP calls are mocked offline):

```bash
pytest tests/
```

---

## Future Improvements

- Add asynchronous HTTP request support using `httpx` or `aiohttp`.
- Support OAuth2 and Bearer Token authentication headers natively.
- Add response output caching using SQLite or Redis.
