# 🧪 Day 48 — Professional Testing with Pytest (E-Commerce Backend V4)

Welcome to **Day 48** of the **Python 200-Day Challenge**! Today we moved from manual API testing to building a **production-grade automated test suite** using **Pytest**, **Pytest-Cov**, **FastAPI `TestClient`**, **Shared Fixtures (`conftest.py`)**, **Parameterized Testing**, **External Gateway Mocking (`unittest.mock`)**, and **Atomic Transaction Rollback Testing**.

---

## 🎯 Day 48 Objectives Accomplished

- [x] **Pytest Automation:** Built an automated test suite containing **50+ meaningful unit & integration tests**.
- [x] **Test Isolation & Fixtures:** Configured isolated SQLite test engine (`test_ecommerce_v4.db`) and Alembic migration runner in `tests/conftest.py`.
- [x] **FastAPI TestClient & Dependency Overrides:** Swapped production database sessions with isolated test database sessions using `app.dependency_overrides`.
- [x] **Security & Parameterized Testing:** Verified Argon2/Bcrypt password hashing, JWT token creation, token expiration, secret validation, and parameterized inputs (`@pytest.mark.parametrize`).
- [x] **Authentication & Role-Based Authorization:** Validated `201 Created` registration, `409 Conflict` duplicate email, `401 Unauthorized` invalid credentials, and `403 Forbidden` admin boundaries (`GET /admin/users`).
- [x] **Atomic Transaction Rollback Testing:** Verified multi-item order placement with stock deduction, and proved atomic rollback (zero partial stock mutations) when item 2 encounters an out-of-stock condition.
- [x] **External Payment Gateway Mocking:** Simulated credit card charges using `unittest.mock.patch` & `MagicMock` (`PaymentGatewayClient.charge`), asserting call parameters and handling gateway decline exceptions.
- [x] **Code Coverage Reporting:** Generated complete code coverage report using `pytest-cov`.

---

## 📁 Repository Structure

```text
Day 48/
├── Day48.md                   # FastAPI & Pytest Professional Testing Masterclass Notes
├── sql_practice/
│   └── testing_practice.sql   # Pure SQL Test Database Setup & Transaction Rollback Practice
├── app/
│   ├── main.py                # FastAPI Application & Composition Root
│   ├── database.py            # SQLAlchemy Engine, SessionLocal & get_db Generator
│   ├── config.py              # Environment Settings (.env)
│   ├── security.py            # Password Hashing (Argon2/Bcrypt) & JWT Utilities
│   ├── exceptions.py          # Custom Domain Exception Taxonomy (400, 401, 403, 404, 409, 502)
│   ├── models/                # ORM Models (User, Product, Order, OrderItem)
│   ├── schemas/               # Pydantic Schemas (Auth, User, Product, Order, Payment)
│   ├── dependencies/          # Security Dependencies (get_current_user, require_admin)
│   ├── repositories/          # Data Access Layer with selectinload
│   ├── services/              # Business Logic (Auth, User, Product, Order, Payment)
│   └── routers/               # APIRouter Modules (Auth, Users, Products, Orders, Payments)
├── alembic/
│   ├── env.py                 # Alembic Migration Environment Script
│   ├── script.py.mako        # Migration Script Template
│   └── versions/              # Revisions (001 through 007_add_auth_fields)
├── tests/                     # 50+ Pytest Automation Test Cases
│   ├── conftest.py            # Test Database Engine, Alembic Fixtures & TestClient
│   ├── test_security.py       # Unit & Parameterized Password/JWT Tests (8 Tests)
│   ├── test_auth.py           # Integration Auth & Token Tests (8 Tests)
│   ├── test_users.py          # Profile & Admin RBAC Tests (6 Tests)
│   ├── test_products.py       # Public Catalog & Admin Mutation Tests (10 Tests)
│   ├── test_orders.py         # Order Checkout & Transaction Rollback Tests (7 Tests)
│   ├── test_payment_mocking.py# Payment Gateway Mocking Tests (4 Tests)
│   ├── test_relationships.py  # Eager Loading & Serialization Tests (3 Tests)
│   └── test_migrations.py     # Alembic Revision & Downgrade Tests (4 Tests)
├── .env.example
├── .env
├── alembic.ini
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🧪 Running the Test Suite & Coverage Report

### 1. Run All Pytest Tests
```bash
pytest "Day 48/tests/"
```

### 2. Run Pytest with Coverage Summary
```bash
pytest "Day 48/tests/" --cov=app --cov-report=term-missing
```

---

## 📊 Summary of Test Coverage

```text
============================= test session starts =============================
collected 50 items

Day 48\tests\test_auth.py ........                                     [ 16%]
Day 48\tests\test_migrations.py ....                                   [ 24%]
Day 48\tests\test_orders.py .......                                     [ 38%]
Day 48\tests\test_payment_mocking.py ....                              [ 46%]
Day 48\tests\test_products.py ..........                                [ 66%]
Day 48\tests\test_relationships.py ...                                  [ 72%]
Day 48\tests\test_security.py .........                                 [ 90%]
Day 48\tests\test_users.py ......                                       [100%]

======================== 50 passed in 1.85s =========================
```
