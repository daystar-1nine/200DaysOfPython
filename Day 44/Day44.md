# 🐍 Day 44/200 – Masterclass Notes: SQL + PostgreSQL + FastAPI Database Integration

🎯 **Goal:** Replace in-memory data arrays with persistent Relational Databases—mastering SQL fundamentals (DDL, DML, JOINs), PostgreSQL database administration, SQLAlchemy 2.0 ORM (`DeclarativeBase`, `Mapped[T]`, `mapped_column()`), database engines, session lifecycle management (`get_db` yield generator), Repository + Database integration, Pydantic Schemas vs SQLAlchemy Models separation, environment variables (`python-dotenv`), and building **User Management API V3**.

---

## 📌 Executive Summary & Key Takeaways

- **Relational Database Core Concepts:**
  - **Database:** Organized container of persistent data tables.
  - **Table (`users`):** Structured grid of records defined by columns and rows.
  - **Primary Key (`id`):** Unique identifier ensuring entity integrity.
  - **Foreign Key (`user_id`):** Column referencing a primary key in another table establishing relationships (`users` 1:N `orders`).
- **Pure SQL Essentials:**
  - `CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(100) NOT NULL, email VARCHAR(255) UNIQUE NOT NULL, age INTEGER);`
  - `INSERT INTO users (name, email, age) VALUES ('Suraj', 'suraj@example.com', 21);`
  - `SELECT name, email FROM users WHERE age > 20 ORDER BY age DESC LIMIT 10;`
  - `SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id;`
- **SQLAlchemy 2.0 ORM:**
  - **SQLAlchemy Model (`app/models/user.py`):** Represents database tables and columns using `Mapped[T]` and `mapped_column()`.
  - **Pydantic Schema (`app/schemas/user.py`):** Represents API HTTP request bodies and JSON responses (`UserCreate`, `UserResponse`).
  - **Session Lifecycle (`get_db()`):** Uses FastAPI `Depends(get_db)` to open a database `Session` per request and close it automatically via `try...finally`.
- **Pushing Filtering to Database:**
  - ❌ **Bad:** Fetching `SELECT * FROM users` into Python and using `filter()` in memory.
  - ✅ **Good:** Executing `select(User).where(User.name.icontains(name))` so PostgreSQL performs indexing and filtering natively.

---

## 📖 Topic 1: SQL Practice Queries 1–10 (With Solutions)

```sql
-- 1. Find all users older than 18
SELECT * FROM users WHERE age > 18;

-- 2. Find users whose name starts with 'S'
SELECT * FROM users WHERE name LIKE 'S%';

-- 3. Sort users by age
SELECT * FROM users ORDER BY age ASC;

-- 4. Get the 5 oldest users
SELECT * FROM users ORDER BY age DESC LIMIT 5;

-- 5. Count users
SELECT COUNT(*) FROM users;

-- 6. Find all orders belonging to user 1
SELECT * FROM orders WHERE user_id = 1;

-- 7. Calculate total order amount
SELECT SUM(amount) AS total_amount FROM orders;

-- 8. Join users and orders
SELECT users.name, orders.amount FROM users JOIN orders ON users.id = orders.user_id;

-- 9. Find users who have orders
SELECT DISTINCT users.name FROM users JOIN orders ON users.id = orders.user_id;

-- 10. Find the highest-value order
SELECT * FROM orders ORDER BY amount DESC LIMIT 1;
```

---

## 📖 Topic 2: SQLAlchemy 2.0 Model vs Pydantic Schema

```python
# 1. SQLAlchemy Database Model (Represents 'users' table in Database)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(nullable=False)
    email: Mapped[str] = mapped_column(unique=True, index=True, nullable=False)
    age: Mapped[int] = mapped_column(nullable=False)

# 2. Pydantic API Schema (Represents HTTP Request / Response JSON)
from pydantic import BaseModel, Field

class UserCreate(BaseModel):
    name: str = Field(..., min_length=2)
    email: str
    age: int = Field(..., gt=0, lt=120)

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    age: int

    class Config:
        from_attributes = True  # Allows ORM model to Pydantic conversion
```

---

## ❓ Challenge Assessment Answers (1–14)

1. **What is a relational database?** A database storing data in structured tables linked by foreign key relationships.
2. **What is a primary key?** A unique column constraint identifying each row in a table.
3. **What is a foreign key?** A column referencing another table's primary key to link records.
4. **What is SQL?** Structured Query Language used to create, query, update, and manage relational databases.
5. **What does SELECT do?** Retrieves data rows and columns from one or more tables.
6. **What does JOIN do?** Combines rows from two or more tables based on a related column between them.
7. **What is an ORM?** Object-Relational Mapper translating database tables into object-oriented Python classes.
8. **What is SQLAlchemy?** The industry-standard Python SQL toolkit and Object-Relational Mapper.
9. **What is a database session?** A transactional workspace holding database connections for queries and commits.
10. **Why should a database session be closed?** To release database connections back to the connection pool and prevent resource leaks.
11. **SQLAlchemy model vs Pydantic schema?** SQLAlchemy models map database table structures; Pydantic schemas validate HTTP API inputs/outputs.
12. **Why store credentials in environment variables?** To keep secret database credentials out of version control systems like Git.
13. **Why let the database handle filtering?** Relational databases use optimized indexes and binary execution engines that process filtering significantly faster than Python loops.
14. **What happens during `POST /users`?** HTTP Request -> Pydantic Validation -> Router -> Service Validation -> Repository -> SQLAlchemy Session -> PostgreSQL Insert -> Pydantic Serialization -> HTTP Response.

---

## ⚡ Master Cheat Sheet

```python
# Database Session Dependency in FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from fastapi import Depends

engine = create_engine("sqlite:///./users.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users")
def list_users(db: Session = Depends(get_db)):
    return db.query(User).all()
```

---

## 📝 Recap Checklist
- [x] Mastered Relational Database concepts (Tables, Primary Keys, Foreign Keys).
- [x] Practiced pure SQL CRUD and JOIN queries (Queries 1–10).
- [x] Defined SQLAlchemy 2.0 ORM models (`DeclarativeBase`, `Mapped[T]`, `mapped_column()`).
- [x] Configured Database Engine, Session factory, and `get_db()` dependency generator.
- [x] Kept Pydantic Schemas separated from SQLAlchemy Models.
- [x] Integrated Environment Variables (`.env`) via `python-dotenv`.
- [x] Built 20+ automated unit & integration test cases.
