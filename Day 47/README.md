# Secure E-Commerce Backend V3 (Authentication, Authorization & Alembic Migrations)

A production-grade, relational e-commerce backend demonstrating **FastAPI Authentication & Authorization**, password hashing with **Argon2 / Bcrypt**, signed **JWT Access Tokens**, **Bearer Authentication**, **Role-Based Access Control (RBAC)** (`user` vs `admin`), **401 Unauthorized vs 403 Forbidden** status codes, **Resource Ownership Isolation**, and **Alembic Database Schema Evolution**.

---

## Overview

`Secure E-Commerce Backend V3` secures all endpoints against unauthenticated access and unauthorized cross-user operations:

```text
                                CLIENT REQUEST
                                      │
                                      ▼
                           POST /auth/login (Email + Password)
                                      │
                                      ▼
                            Argon2/Bcrypt Hash Verification
                                      │
                                      ▼
                            Issue Signed JWT Access Token
                                      │
                                      ▼
             ┌────────────────────────┴────────────────────────┐
             ▼                                                 ▼
     AUTHENTICATED USER                                    ADMIN USER
(Authorization: Bearer <token>)                   (Authorization: Bearer <token>)
  • GET /users/me                                   • GET /admin/users
  • GET /users/me/orders                            • POST /products
  • POST /orders                                    • PATCH /products/{id}
  • GET /orders/{id} (Own Only)                     • DELETE /products/{id}
                                                    • GET /orders/{id} (All)
```

---

## Features

- 🔐 **Argon2 / Bcrypt Password Hashing:** Hashes passwords with one-way salt digests before storing in database (`password_hash`).
- 🔑 **Cryptographic JWT Access Tokens:** Generates HMAC-SHA256 signed access tokens with expiration (`exp` claim).
- 🛡️ **Role-Based Access Control (RBAC):** Restricts admin endpoints (`POST /products`, `GET /admin/users`) with HTTP 403 Forbidden checks.
- 🔒 **Resource Ownership Isolation:** Protects customer orders on `GET /orders/{id}` and `GET /users/me/orders` so normal users cannot view other users' orders.
- ⚡ **Alembic Schema Evolution:** Version-controlled database schema migration `007_add_auth_fields.py` adding `password_hash`, `role`, and `age` columns.
- 🧪 **35+ Automated Test Cases:** Comprehensive test suite validating registration, login, password hashing, valid/invalid JWTs, expiration, 401/403 errors, and database migration rollbacks.

---

## Alembic Migration History

| Revision | Description | Changes Applied |
| :--- | :--- | :--- |
| `001` | `create initial tables` | Base tables: `users`, `products`, `orders`, `order_items` |
| `002` | `add user phone column` | Added `phone` column to `users` |
| `003` | `add product category column` | Added `category` column to `products` |
| `004` | `add product created_at column` | Added `created_at` column to `products` |
| `005` | `add indexes` | Created `idx_users_email` (unique) and `idx_products_name` |
| `006` | `add user created_at with data migration` | Added `created_at` to `users` with default values for existing rows |
| `007` | `add authentication fields` | Added `password_hash`, `role`, and `age` to `users` |

---

## Project Structure

```text
Day 47/
├── Day47.md                      # Masterclass Notes & Interview Q&A
├── sql_practice/
│   └── auth_practice.sql        # Pure SQL Practice for Hashes, Roles & Ownership Queries
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI Composition Root
│   ├── database.py                # Engine, SessionLocal, Base, get_db Generator
│   ├── config.py                  # Environment Settings (.env)
│   ├── security.py                # Password Hashing & JWT Token Utilities
│   ├── exceptions.py              # Custom Exception Taxonomy (401, 403, 404)
│   ├── models/                    # ORM Models (user, product, order, order_item)
│   ├── schemas/                   # Pydantic Schemas (auth, user, product, order)
│   ├── dependencies/              # FastAPI Dependencies (get_current_user, require_admin)
│   ├── repositories/              # Data Access Layer with selectinload
│   ├── services/                  # Business Services (auth_service, order_service)
│   └── routers/                   # APIRouter Modules (auth, users, products, orders)
├── alembic/
│   ├── env.py                     # Alembic Environment Script
│   ├── script.py.mako            # Migration template
│   └── versions/                  # Revision Scripts (001 through 007)
├── tests/                         # Test Suite (35+ Test Cases)
│   ├── conftest.py
│   ├── test_auth.py
│   ├── test_users.py
│   ├── test_products.py
│   ├── test_orders.py
│   ├── test_relationships.py
│   └── test_migrations.py
├── .env.example
├── .env
├── alembic.ini                    # Alembic Global Configuration
├── requirements.txt               # Dependencies
├── pyproject.toml                 # Package Metadata
├── .gitignore
└── README.md                      # Documentation
```

---

## Installation

```bash
# Navigate to Day 47 directory
cd Day\ 47

# Create virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running Alembic Migrations

```bash
# Apply all 7 migrations to latest head version
alembic upgrade head

# Roll back most recent migration
alembic downgrade -1
```

---

## Running the API Server

```bash
uvicorn app.main:app --reload --port 8000
```

Open interactive Swagger UI docs:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

---

## Testing

Run complete Pytest test suite (35+ tests):

```bash
pytest tests/
```
