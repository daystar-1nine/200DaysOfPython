# User Management API V3 (SQL + PostgreSQL + FastAPI Database Integration)

A persistent, enterprise-grade REST API demonstrating **FastAPI Integration with Relational Databases (PostgreSQL / SQLite)** using **SQLAlchemy 2.0 ORM**, **Pydantic v2**, and **Layered Backend Architecture**.

---

## Overview

`User Management API V3` replaces transient in-memory arrays with persistent database tables:

```text
Request ──> Router (HTTP) ──> Service (Validation) ──> Repository (Data Access) ──> SQLAlchemy 2.0 ──> PostgreSQL / SQLite Database
```

---

## Features

- 🗄️ **Relational Database Persistence:** Stores records in persistent database tables via SQLAlchemy 2.0 ORM (`DeclarativeBase`, `Mapped[T]`, `mapped_column()`).
- ⚡ **Database-Level Search & Filtering:** Pushes string search (`WHERE name ILIKE '%query%'`) directly down to the database engine.
- 💉 **Session Lifecycle Dependency:** Injects `get_db()` session generator into FastAPI endpoints, guaranteeing clean session open & close per request (`try...finally`).
- 🔒 **Environment Variable Credentials:** Loads `DATABASE_URL` safely from `.env` using `python-dotenv`.
- 📐 **Separation of Schemas & Models:** Clean distinction between Pydantic API schemas (`UserCreate`, `UserResponse`) and SQLAlchemy database ORM models (`User`).
- 📊 **Automated Test Suite:** 25 unit & integration test cases testing UserRepository, SQLAlchemy Session, and REST endpoints.

---

## Tech Stack

- **Framework:** FastAPI
- **ORM:** SQLAlchemy 2.0
- **Databases Supported:** PostgreSQL (`psycopg3`), SQLite (`sqlite3`)
- **Environment Management:** `python-dotenv`
- **Data Validation:** Pydantic v2
- **Testing:** Pytest, HTTPX, FastAPI TestClient

---

## Project Structure

```text
Day 44/
├── Day44.md                      # Masterclass Notes
├── sql_practice/
│   └── sql_practice.sql           # Pure SQL Queries 1-10
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI Composition Root (Base.metadata.create_all)
│   ├── database.py                # Engine, SessionLocal, Base, get_db Generator
│   ├── config.py                  # Environment Settings (.env)
│   ├── exceptions.py              # Domain Exceptions Taxonomy
│   ├── models/
│   │   └── user.py                # SQLAlchemy 2.0 ORM Model
│   ├── schemas/
│   │   └── user.py                # Pydantic Schemas
│   ├── repositories/
│   │   └── user_repository.py     # Data Access Layer (SQLAlchemy Session Queries)
│   ├── services/
│   │   └── user_service.py        # Business Logic Layer
│   └── routers/
│       └── users.py               # User REST API Endpoints
├── tests/                         # Test Suite (25 Test Cases)
│   ├── test_repository.py
│   └── test_users.py
├── .env.example
├── .env
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Installation

```bash
# Navigate to Day 44 project directory
cd Day\ 44

# Create virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Environment Configuration

Create a `.env` file in the project root:

```env
DATABASE_URL=sqlite:///./users_v3.db
# For PostgreSQL:
# DATABASE_URL=postgresql+psycopg://username:password@localhost:5432/python_learning
ENVIRONMENT=development
DEBUG=True
```

---

## Running the API

Start Uvicorn server:

```bash
uvicorn app.main:app --reload --port 8000
```

Open interactive API docs:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

---

## API Endpoints

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| **GET** | `/` | Root Welcome Endpoint |
| **GET** | `/health` | System & Database Health Check |
| **GET** | `/about` | API Metadata |
| **GET** | `/users` | Paginated User Collection (`skip`, `limit`) |
| **GET** | `/users/search` | Database ILIKE Search by name (`WHERE name ILIKE '%query%'`) |
| **GET** | `/users/{id}` | Get User by ID |
| **POST** | `/users` | Create User (201 Created with duplicate email 409 check) |
| **PUT** | `/users/{id}` | Replace User in database |
| **PATCH** | `/users/{id}` | Partially update User in database |
| **DELETE** | `/users/{id}` | Delete User from database |

---

## Testing

Run complete Pytest test suite:

```bash
pytest tests/
```
