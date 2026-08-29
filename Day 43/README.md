# User Management API V2

A production-grade, multi-layered REST API demonstrating **FastAPI Architecture**, **APIRouter** modularization, and **Dependency Injection** (`Depends()`).

---

## Overview

`User Management API V2` refactors basic API handlers into a clean, scalable enterprise architecture:

```text
Request ──> Router (HTTP) ──> Dependencies (Auth/Config) ──> Service (Validation) ──> Repository (Data Access) ──> Data
```

---

## Features

- 🏗️ **Layered Architecture:** Clear separation of concerns between Router, Service, and Repository layers.
- 💉 **FastAPI Dependency Injection:** Uses `Depends()` to inject repositories, services, auth, and configuration settings dynamically.
- 🔐 **Protected Auth Endpoints:** Protected `/profile` endpoint utilizing `Depends(get_current_user)`.
- 🔀 **Multi-Router Support:** Modular `users_router` and `products_router` integrated cleanly in `main.py`.
- 🧪 **Dependency Overrides in Testing:** Demonstrates `app.dependency_overrides` for dependency isolation during unit testing.
- 📊 **Comprehensive Test Suite:** 32 unit & integration test cases testing Repository, Service, API, and Overrides.

---

## Tech Stack

- **Framework:** FastAPI
- **ASGI Server:** Uvicorn
- **Data Validation:** Pydantic v2
- **Testing:** Pytest, HTTPX, FastAPI TestClient

---

## Project Structure

```text
Day 43/
├── exercises/                   # Independent Practice Exercises 1-4
│   ├── ex1_user_injection.py
│   ├── ex2_settings_injection.py
│   ├── ex3_service_repo_connection.py
│   └── ex4_products_router.py
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI Composition Root
│   ├── config.py                # Configuration Settings Dependency
│   ├── exceptions.py            # Custom Exception Hierarchy & Handlers
│   ├── models/
│   │   ├── user.py              # Pydantic Schemas
│   │   └── product.py
│   ├── repositories/
│   │   ├── user_repository.py   # Data Access Layer
│   │   └── product_repository.py
│   ├── services/
│   │   ├── user_service.py      # Business Logic Layer
│   │   └── product_service.py
│   ├── dependencies/
│   │   ├── auth.py              # Authentication Dependencies
│   │   └── providers.py         # Service & Repository Factory Providers
│   └── routers/
│       ├── users.py             # User & Profile Endpoints Router
│       └── products.py          # Product Endpoints Router
├── tests/                       # Complete Test Suite (32 Test Cases)
│   ├── test_repository.py
│   ├── test_services.py
│   ├── test_users.py
│   └── test_overrides.py
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Installation

```bash
# Navigate to Day 43 project directory
cd Day\ 43

# Create virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the API

Start Uvicorn server:

```bash
uvicorn app.main:app --reload --port 8000
```

Open interactive Swagger UI docs:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

---

## API Endpoints

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| **GET** | `/` | Root Welcome Endpoint |
| **GET** | `/health` | System Health Check |
| **GET** | `/about` | API Metadata |
| **GET** | `/config` | Configuration Settings (Injected via `Depends(get_settings)`) |
| **GET** | `/profile` | Protected User Profile (Injected via `Depends(get_current_user)`) |
| **GET** | `/users` | Paginated User Collection (`skip`, `limit`) |
| **GET** | `/users/search` | Search Users by name query parameter |
| **GET** | `/users/{id}` | Get User by ID |
| **POST** | `/users` | Create User (201 Created) |
| **PUT** | `/users/{id}` | Replace User completely |
| **PATCH** | `/users/{id}` | Update User attributes partially |
| **DELETE** | `/users/{id}` | Delete User by ID |
| **GET** | `/products` | List Products (Multi-Router Component) |

---

## Testing

Run complete Pytest test suite:

```bash
pytest tests/
```

---

## Future Improvements

- Add asynchronous database support with SQLAlchemy 2.0 & AsyncSession.
- Add JWT token parsing inside `get_current_user` auth dependency.
- Integrate Redis caching for repository queries.
