# 🐍 Day 45/200 – Masterclass Notes: Database Relationships & Advanced SQLAlchemy

🎯 **Goal:** Master relational database modeling in SQLAlchemy 2.0—exploring One-to-One, One-to-Many, and Many-to-Many (Association Table) relationships, `ForeignKey` vs `relationship(back_populates=...)`, Eager (`selectinload()`, `joinedload()`) vs Lazy loading strategies, solving the **N+1 Query Problem**, cascading deletes (`cascade="all, delete-orphan"`), atomic database transactions (`BEGIN`, `COMMIT`, `ROLLBACK`), nested Pydantic responses, and building a production **Mini E-Commerce Backend**.

---

## 📌 Executive Summary & Key Takeaways

- **Relationship Types Overview:**
  - **One-to-One (1:1):** `User` ↔ `Profile` (e.g. `uselist=False`).
  - **One-to-Many (1:N):** `User` 1 ↔ N `Order` (`User.orders` holds `list[Order]`, `Order.user_id` is a `ForeignKey("users.id")`).
  - **Many-to-Many (N:M):** `Student` N ↔ M `Course` (Requires junction table `student_courses` holding `student_id` and `course_id`).
- **`ForeignKey` vs `relationship()`:**
  - **`ForeignKey("users.id")`:** Database-level DDL constraint declaring a foreign key link between tables.
  - **`relationship("Order", back_populates="user")`:** Python ORM-level property enabling bidirectional navigation between Python object instances (`user.orders` and `order.user`).
- **Eager Loading & The N+1 Problem:**
  - **N+1 Problem:** Querying 100 users (`1` query) and accessing `user.orders` lazily executes `100` additional queries (Total: `101` queries).
  - **Solution (`selectinload(User.orders)`):** Executes `2` queries total (`SELECT * FROM users` + `SELECT * FROM orders WHERE user_id IN (1,2,...)`).
- **Database Transactions & Atomic Operations:**
  - Order creation must be **atomic**. If reducing stock for Product #2 fails due to insufficient inventory, all preceding stock updates and order item creations must be **rolled back** (`db.rollback()`) to guarantee zero partial database corruptions.

---

## 📖 Topic 1: Tasks 1–7 SQL Practice & Answers

```sql
-- 1. Create Users and Orders Tables
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    amount NUMERIC(10, 2) NOT NULL
);

-- 2. Query User Orders with JOIN
SELECT users.name, orders.amount 
FROM users 
JOIN orders ON users.id = orders.user_id;

-- 3. Calculate Total Spending per User (SUM & GROUP BY)
SELECT users.name, SUM(orders.amount) AS total_spent
FROM users
JOIN orders ON users.id = orders.user_id
GROUP BY users.id, users.name;

-- 7. Many-to-Many Relationship Schema & Queries
CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

CREATE TABLE courses (
    id SERIAL PRIMARY KEY,
    title VARCHAR(100) NOT NULL
);

CREATE TABLE student_courses (
    student_id INTEGER REFERENCES students(id) ON DELETE CASCADE,
    course_id INTEGER REFERENCES courses(id) ON DELETE CASCADE,
    PRIMARY KEY (student_id, course_id)
);

-- Query: All courses taken by Suraj
SELECT courses.title 
FROM courses 
JOIN student_courses ON courses.id = student_courses.course_id
JOIN students ON students.id = student_courses.student_id
WHERE students.name = 'Suraj';
```

---

## 📖 Topic 2: SQLAlchemy 2.0 Bidirectional One-to-Many Modeling

```python
from typing import List, Optional
from sqlalchemy import String, ForeignKey, Numeric
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(255), unique=True)

    # 1:N Relationship to Order
    orders: Mapped[List["Order"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan"
    )

class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    amount: Mapped[float] = mapped_column(Numeric(10, 2))

    # N:1 Back Relationship to User
    user: Mapped["User"] = relationship(back_populates="orders")
```

---

## ❓ Interview Questions & Answers (1–15)

1. **What is a foreign key?** A database table constraint linking a column to the primary key of another table.
2. **One-to-one vs one-to-many?** 1:1 links one entity instance to exactly one other (User-Profile); 1:N links one entity instance to multiple related instances (User-Orders).
3. **What is many-to-many?** A relationship where multiple records in Table A link to multiple records in Table B (Students-Courses).
4. **Why do we need an association table?** Relational databases cannot store array lists of IDs inside a single cell, so a junction table with compound foreign keys resolves N:M relationships.
5. **`ForeignKey` vs `relationship()`?** `ForeignKey` enforces database DDL constraints; `relationship()` provides Python ORM navigation attributes.
6. **What is `back_populates`?** Establishes bidirectional synchronicity between relationship properties on both model classes.
7. **What is lazy loading?** Deferring the loading of related child objects until they are explicitly accessed in Python code.
8. **What is eager loading?** Fetching parent and related child objects simultaneously in optimized initial SQL queries (`selectinload`).
9. **What is the N+1 problem?** Executing 1 query for N parent items and then N additional queries for each item's children.
10. **How to reduce N+1 queries?** Use SQLAlchemy `selectinload()` or `joinedload()` eager loading.
11. **What is a database transaction?** A logical unit of work containing multiple SQL operations executed under ACID guarantees.
12. **`COMMIT` vs `ROLLBACK`?** `COMMIT` permanently saves all transaction changes; `ROLLBACK` aborts the transaction and reverts all pending modifications.
13. **What happens if an order item is out of stock?** The service raises an `InsufficientStockError`, the database transaction rolls back, and no partial order is created.
14. **Why should order creation be transactional?** To prevent inconsistent states like stock being decremented without creating order items or vice versa.
15. **SQLAlchemy model vs Pydantic schema?** SQLAlchemy models represent database tables and ORM instances; Pydantic schemas validate API request payloads and structure JSON responses.

---

## ⚡ Master Cheat Sheet

```python
# Eager Loading in SQLAlchemy 2.0
from sqlalchemy import select
from sqlalchemy.orm import selectinload

# Prevents N+1 query problem by loading orders in a single IN (...) query
stmt = select(User).options(selectinload(User.orders)).where(User.id == user_id)
user = db.scalars(stmt).first()
```

---

## 📝 Recap Checklist
- [x] Understood 1:1, 1:N, and N:M database relationship cardinalities.
- [x] Implemented `ForeignKey` DDL constraints and `relationship(back_populates=...)` ORM properties.
- [x] Mastered `selectinload()` eager loading to eliminate the N+1 query problem.
- [x] Implemented atomic database transactions (`COMMIT` / `ROLLBACK`) for stock management.
- [x] Built nested Pydantic response models (`UserWithOrdersResponse`).
- [x] Built 25+ automated Pytest unit and integration tests.
