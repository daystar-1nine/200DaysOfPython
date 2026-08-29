# User Management API

A production-style, fully documented **User Management REST API** built with **FastAPI**, **Pydantic**, and **Uvicorn**.

---

## Overview

`User Management API` provides complete RESTful CRUD operations over user resource entities, including pagination, keyword searching, Pydantic `BaseModel` & `Field` request/response validation, custom exception mapping, and automatic OpenAPI interactive documentation (`/docs` & `/redoc`).

---

## Features

- ⚡ **FastAPI High Performance:** Built on Starlette and Pydantic with standard Python type annotations.
- 📖 **Automatic Documentation:** Interactive Swagger UI documentation rendered automatically at `/docs` and ReDoc at `/redoc`.
- 🔐 **Pydantic Data Validation:** Enforces string length constraints (`min_length=2`) and numerical bounds (`gt=0, lt=120`) using Pydantic `Field`.
- 📄 **Pagination & Search:** Supports list pagination (`/users?skip=0&limit=10`) and keyword search (`/users/search?name=suraj`).
- 🛡️ **Structured Error Responses:** Domain exception handlers returning HTTP status codes (`404 Not Found`, `409 Conflict`, `422 Validation Error`).
- 🧪 **Comprehensive Test Suite:** 19 automated unit & integration tests built with FastAPI `TestClient`.

---

## Tech Stack

- **Framework:** FastAPI
- **ASGI Server:** Uvicorn
- **Data Validation:** Pydantic v2
- **Testing:** Pytest, HTTPX, FastAPI TestClient

---

## Project Structure

```text
Day 42/
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI App Composition Root (/health, /about)
│   ├── exceptions.py       # Domain Exceptions & FastAPI Handlers
│   ├── models/
│   │   ├── __init__.py
│   │   └── user.py         # Pydantic Schemas (UserCreate, UserUpdate, UserPatch, UserResponse)
│   ├── routes/
│   │   ├── __init__.py
│   │   └── users.py        # APIRouter for /users endpoints
│   └── services/
│       ├── __init__.py
│       └── user_service.py # UserService (In-memory storage & business logic)
├── tests/
│   ├── test_users.py       # FastAPI TestClient API Tests
│   └── test_validation.py  # Pydantic Schema Validation Tests
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Installation

```bash
# Navigate to Day 42 project directory
cd Day\ 42

# Create virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the API

Start the Uvicorn ASGI server with hot-reload enabled:

```bash
uvicorn app.main:app --reload --port 8000
```

Open interactive Swagger UI docs in your browser:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

---

## API Endpoints

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| **GET** | `/` | Root Welcome Endpoint |
| **GET** | `/health` | System Health Check (`{"status": "healthy"}`) |
| **GET** | `/about` | API Metadata & Environment Info |
| **GET** | `/users` | List Users (with `skip` and `limit` pagination) |
| **GET** | `/users/search` | Search Users by name query parameter |
| **GET** | `/users/{id}` | Get User by unique integer ID |
| **POST** | `/users` | Create User (201 Created) |
| **PUT** | `/users/{id}` | Replace User completely (PUT semantics) |
| **PATCH** | `/users/{id}` | Partially update User attributes (PATCH semantics) |
| **DELETE** | `/users/{id}` | Delete User by ID |

---

## Request & Response Examples

### 1. Create User (`POST /users`)

**Request Payload:**
```json
{
  "name": "Suraj Sawant",
  "email": "suraj@example.com",
  "age": 21
}
```

**Response (`201 Created`):**
```json
{
  "id": 5,
  "name": "Suraj Sawant",
  "email": "suraj@example.com",
  "age": 21
}
```

---

## Testing

Run complete Pytest test suite:

```bash
pytest tests/
```

---

## Future Improvements

- Integrate persistent database storage with SQLAlchemy & SQLite / PostgreSQL.
- Add JWT (JSON Web Tokens) Authentication & Role-Based Access Control (RBAC).
- Add asynchronous database queries with `async def` route handlers.
