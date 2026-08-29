# Mini E-Commerce Backend (Advanced SQLAlchemy Relationships & FastAPI)

A production-grade, relational e-commerce backend demonstrating **Advanced SQLAlchemy 2.0 Relationships (1:1, 1:N, N:M)**, **Foreign Keys**, **Eager Loading (`selectinload`)**, **N+1 Query Prevention**, **Atomic Database Transactions**, and **Nested Pydantic Schemas**.

---

## Overview

`Mini E-Commerce Backend` links relational domain entities using clean SQLAlchemy 2.0 ORM models:

```text
User (1) ──< Orders (N) ──< OrderItems (N) >── (1) Product
```

---

## Features

- 🔗 **Relational Data Modeling:** Establishes 1:N relationships (`User` ↔ `Order`, `Order` ↔ `OrderItem`, `Product` ↔ `OrderItem`) with `ForeignKey` and `relationship(back_populates=...)`.
- ⚡ **Eager Loading & N+1 Prevention:** Uses `selectinload(User.orders)` to fetch nested child collections in single batch `IN` queries.
- 💳 **Atomic Order Placement Transactions:** 8-step checkout transaction verifying stock availability, deducting product inventory, creating line items, and executing `COMMIT` or `ROLLBACK` atomically.
- 🗑️ **Cascading Deletes:** Configures `cascade="all, delete-orphan"` so deleting a user automatically cleans up child orders.
- 📐 **Nested JSON Schemas:** Returns nested Pydantic responses (`UserWithOrdersResponse`) containing lists of embedded orders and line items.
- 📊 **Comprehensive Test Suite:** 26 automated unit & integration test cases testing relationships, stock deductions, transactions, and API endpoints.

---

## Tech Stack

- **Framework:** FastAPI
- **ORM:** SQLAlchemy 2.0
- **Databases Supported:** PostgreSQL (`psycopg3`), SQLite (`sqlite3`)
- **Data Validation:** Pydantic v2
- **Testing:** Pytest, HTTPX, FastAPI TestClient

---

## Project Structure

```text
Day 45/
├── Day45.md                      # Masterclass Notes & Interview Q&A
├── sql_practice/
│   └── relationships_practice.sql # Pure SQL Relationships & Joins Practice
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI Composition Root (Base.metadata.create_all)
│   ├── database.py                # Engine, SessionLocal, Base, get_db Generator
│   ├── config.py                  # Environment Settings (.env)
│   ├── exceptions.py              # ECommerce Exception Taxonomy
│   ├── models/
│   │   ├── user.py                # User ORM Model (1:N Orders)
│   │   ├── product.py             # Product ORM Model
│   │   ├── order.py                # Order ORM Model (N:1 User, 1:N OrderItems)
│   │   └── order_item.py           # OrderItem ORM Junction Model
│   ├── schemas/
│   │   ├── user.py                # User Pydantic Schemas (UserWithOrdersResponse)
│   │   ├── product.py             # Product Pydantic Schemas
│   │   └── order.py               # Order & OrderItem Pydantic Schemas
│   ├── repositories/
│   │   ├── user_repository.py     # UserRepository with selectinload
│   │   ├── product_repository.py  # ProductRepository
│   │   └── order_repository.py    # OrderRepository with selectinload
│   ├── services/
│   │   ├── user_service.py        # UserService
│   │   ├── product_service.py     # ProductService
│   │   └── order_service.py       # Transactional Order Placement Engine
│   └── routers/
│       ├── users.py               # User Endpoints (/users/{id}/orders)
│       ├── products.py            # Product Endpoints
│       └── orders.py              # Order Endpoints (POST /orders)
├── tests/                         # Test Suite (26 Test Cases)
│   ├── test_users.py
│   ├── test_products.py
│   ├── test_orders.py
│   └── test_relationships.py
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
# Navigate to Day 45 project directory
cd Day\ 45

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
| **GET** | `/health` | System & Database Health Check |
| **GET** | `/users` | List Users |
| **GET** | `/users/{id}` | Get User by ID |
| **GET** | `/users/{id}/with-orders` | Get User with Nested Orders |
| **GET** | `/users/{id}/orders` | Get User Orders Collection |
| **POST** | `/users` | Create User |
| **GET** | `/products` | List Products |
| **GET** | `/products/{id}` | Get Product by ID |
| **POST** | `/products` | Create Product with stock |
| **PATCH** | `/products/{id}` | Update Product stock or price |
| **DELETE** | `/products/{id}` | Delete Product |
| **POST** | `/orders` | Atomic Order Checkout Transaction |
| **GET** | `/orders` | List Orders |
| **GET** | `/orders/{id}` | Get Order Details with line items |

---

## Testing

Run complete Pytest test suite:

```bash
pytest tests/
```
