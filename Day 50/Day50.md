# 🐍 Day 50 — The First Big Milestone (Revision + Assessment + Project Polish)

> **"Day 50 is not about learning another random topic. Today is about consolidating everything from Day 1 to Day 49, testing yourself like a real backend engineer, cleaning your backend project architecture, and completing a production-ready milestone project."**

---

## 🗺️ Day 50 Architecture & Roadmap

```text
                                   DAY 50
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       ↓                              ↓                              ↓
  PYTHON REVISION              BACKEND REVISION                THEORY ASSESSMENT
  (Days 1–40)                  (Days 41–49)                    (25 Core Q&As)
       │                              │                              │
       └──────────────────────────────┼──────────────────────────────┘
                                      ↓
                              CODING CHALLENGES
                              (Challenges 1 to 5)
                                      ↓
                              MILESTONE PROJECT
                              (TaskFlow API)
                                      ↓
                                GITHUB UPDATE
```

---

## 📚 PART 1 — Comprehensive Python Revision (Days 1–40)

### 1. Variables & Data Types
Python is dynamically typed and strongly typed. Primitive data types include `int`, `float`, `str`, `bool`, and `NoneType`. Collections include `list`, `tuple`, `set`, and `dict`.

```python
# Type hierarchy and mutability overview:
# Mutable: list, set, dict
# Immutable: int, float, str, tuple, bool, NoneType
```

### 2. Control Flow & Conditions
Conditional execution uses `if`, `elif`, and `else`. Short-circuit logic evaluates boolean expressions (`and`, `or`, `not`) efficiently.

### 3. Loops & Iteration
- `for` loops iterate over sequence iterables using Python's iterator protocol (`__iter__` and `__next__`).
- `while` loops repeat execution based on truthy conditional evaluation.
- Control directives `break` (exit loop) and `continue` (skip iteration) modify default loop flow.

### 4. Functions & Scope
Functions encapsulate reusable logic. Python scopes follow the **LEGB rule** (Local, Enclosing, Global, Built-in). Functions support positional, keyword, default arguments, variadic positional (`*args`), and variadic keyword (`**kwargs`).

### 5. Data Collections & Memory Trade-offs
- **List**: Ordered, mutable sequence with indexing $O(1)$ and lookup $O(n)$.
- **Tuple**: Ordered, immutable sequence with lower memory overhead and hashability.
- **Set**: Unordered collection of unique hashable elements with $O(1)$ lookup via hashtable.
- **Dictionary**: Key-value mapping preserving insertion order (Python 3.7+), with $O(1)$ key lookup.

### 6. Comprehensions
List, Dict, Set, and Generator comprehensions provide declarative, single-line data transformations:
- List: `[x**2 for x in items if x > 0]`
- Dict: `{k: v for k, v in pairs}`
- Set: `{x for x in items}`
- Generator: `(x**2 for x in items)` (lazy evaluation)

### 7. Exception Handling & Robustness
`try...except...else...finally` allows graceful error interception:
- `try`: Code block monitored for exceptions.
- `except`: Catches and handles specific exception instances.
- `else`: Executes only when no exception occurs.
- `finally`: Guarantees cleanup execution (e.g., closing file descriptors or connections).

### 8. Object-Oriented Programming (OOP)
- **Encapsulation**: Bundling state and behaviors while restricting direct internal access using single (`_protected`) or double (`__private`) underscores.
- **Inheritance**: Subclassing base classes to inherit attributes and methods while overriding specific behaviors via `super()`.
- **Polymorphism**: Interfacing with different types through a unified contract (duck typing / protocols).
- **Abstraction**: Defining abstract contracts using `abc.ABC` and `@abstractmethod`.

### 9. Advanced Python Concepts
- **Decorators**: Higher-order functions that wrap another function to extend behavior dynamically.
- **Generators**: Functions yielding values lazily using `yield`, conserving RAM for large stream processing.
- **Context Managers**: Protocols implementing `__enter__` and `__exit__` (or `@contextmanager`) for deterministic resource management.
- **Type Hints & Dataclasses**: Providing static type analysis contracts (`typing`, `dataclass`) to improve code safety and IDE auto-completion.

---

## 🌐 PART 2 — Backend & FastAPI Revision (Days 41–49)

### 10. HTTP Protocol & REST Standards
- **Client-Server Architecture**: Stateless request-response cycle over TCP/IP.
- **Methods**:
  - `GET`: Idempotent & safe data retrieval.
  - `POST`: Non-idempotent resource creation.
  - `PUT`: Idempotent full resource replacement.
  - `PATCH`: Idempotent partial resource update.
  - `DELETE`: Idempotent resource removal.
- **Status Codes**:
  - `200 OK`, `201 Created`, `204 No Content`
  - `400 Bad Request`, `401 Unauthorized`, `403 Forbidden`, `404 Not Found`, `409 Conflict`, `422 Unprocessable Entity`
  - `500 Internal Server Error`

### 11. FastAPI Framework Core
FastAPI builds on top of **Starlette** (ASGI server framework) and **Pydantic** (data validation and serialization):
- **Automatic Validation**: Pydantic validates incoming JSON body, path params, and query params.
- **OpenAPI / Swagger**: Interactive documentation generated automatically at `/docs` and `/redoc`.
- **Dependency Injection**: `Depends()` manages shared session lifecycles, authentication, settings, and services seamlessly.

### 12. SQL & Relational Databases (PostgreSQL / SQLite)
Relational databases enforce schema integrity, constraints (`PRIMARY KEY`, `FOREIGN KEY`, `UNIQUE`), and ACID transactional guarantees.

### 13. SQLAlchemy ORM (2.0 Unified Paradigm)
- **Engine & Session**: Manages database connection pools and unit-of-work state tracking.
- **Declarative Models**: `Mapped[T]` and `mapped_column()` map Python attributes to database columns.
- **Relationships**: `relationship(back_populates=...)` defines ORM entity links, leveraging eager loading (`selectinload`) to eliminate $N+1$ query performance bugs.

### 14. Database Migrations with Alembic
Alembic tracks database schema versioning over time via revision scripts, allowing safe schema evolution (`upgrade`) and rollback (`downgrade`) without manual SQL edits.

### 15. Security & Authentication
- **Password Hashing**: Storing passwords safely using one-way bcrypt hashing with automatic salting (Passlib).
- **JSON Web Tokens (JWT)**: Stateless authentication via signed tokens containing user claims (`sub`, `exp`, `role`) passed in `Authorization: Bearer <token>` headers.
- **Authorization (RBAC)**: Role-Based Access Control enforcing permission boundaries (e.g., Admin vs User).

### 16. Observability, Logging & Error Handling
- **Structured JSON Logging**: Formatting log records as machine-readable JSON objects with ISO timestamps, log levels, and sensitive parameter masking.
- **ASGI Middleware**: Intercepting requests for distributed tracing (`X-Request-ID`) and execution latency header insertion (`Process-Time-Ms`).
- **Health Probes**: `GET /health` (liveness) and `GET /health/ready` (readiness with `SELECT 1` database ping).

---

## 📝 PART 3 — Theory Assessment (25 Core Questions & Answers)

### Q1: What is the difference between list, tuple, set, and dictionary?
- **List**: Ordered, mutable sequence of items allowing duplicates (`[1, 2, 2]`).
- **Tuple**: Ordered, immutable sequence of items allowing duplicates (`(1, 2, 2)`).
- **Set**: Unordered collection of unique, hashable elements (`{1, 2}`).
- **Dictionary**: Key-value mapping with unique, hashable keys (`{"a": 1}`).

### Q2: What is the difference between `==` and `is`?
- `==` checks for **value equality** (whether contents of two objects are equal).
- `is` checks for **identity equality** (whether two references point to the exact same object in memory address `id(obj)`).

### Q3: What is mutable vs immutable?
- **Mutable**: Object state can be modified after creation without changing memory identity (e.g., `list`, `dict`, `set`).
- **Immutable**: Object state cannot be modified; any modification creates a new object in memory (e.g., `int`, `str`, `tuple`, `bool`).

### Q4: What does `yield` do?
`yield` turns a normal Python function into a **generator function**. Instead of returning a complete list at once, `yield` returns values lazily one at a time, preserving execution state between iterations and optimizing memory usage.

### Q5: What is a decorator?
A decorator is a callable that takes another function as an argument, extends or modifies its behavior without altering its original source code, and returns the modified function wrapper.

### Q6: What is the difference between `*args` and `**kwargs`?
- `*args` captures arbitrary non-keyword (positional) arguments into a `tuple`.
- `**kwargs` captures arbitrary keyword arguments into a `dict`.

### Q7: What is the difference between shallow copy and deep copy?
- **Shallow Copy** (`copy.copy()`): Creates a new outer object, but copies references to child nested objects.
- **Deep Copy** (`copy.deepcopy()`): Recursively creates new objects for both the outer container and all nested child objects.

### Q8: What is exception handling?
Exception handling is a mechanism (`try...except...finally`) that catches runtime errors, prevents application crashes, logs diagnostic tracebacks, and allows application recovery or graceful degradation.

### Q9: What is a virtual environment?
A virtual environment is an isolated Python execution runtime directory containing its own Python binary and independent set of installed third-party packages, preventing dependency version conflicts across projects.

### Q10: What is the difference between a class and an object?
- **Class**: A blueprint or schema defining attributes (state) and methods (behavior).
- **Object**: A concrete instance created in memory based on a class blueprint.

### Q11: What is REST?
REST (Representational State Transfer) is an architectural style for designing networked web APIs utilizing stateless communication, standardized HTTP methods (GET, POST, PUT, DELETE), and standardized resource URIs.

### Q12: What is the difference between PUT and PATCH?
- **PUT**: Replaces the entire resource payload. Unspecified fields in the request body are typically reset or removed.
- **PATCH**: Applies partial updates to specified fields of a resource without modifying unspecified fields.

### Q13: What does HTTP 401 mean?
`401 Unauthorized` means the client is unauthenticated (missing or invalid credentials/token) and must authenticate before accessing the resource.

### Q14: What does HTTP 403 mean?
`403 Forbidden` means the client is authenticated, but lacks sufficient permissions (roles/rights) to access the requested resource.

### Q15: What is Pydantic used for?
Pydantic is a data validation and settings management library using Python type annotations. It parses, validates, and serializes request bodies, query parameters, and API response payloads.

### Q16: What is SQLAlchemy?
SQLAlchemy is Python's industry-standard SQL Toolkit and Object-Relational Mapper (ORM), providing high-level Python abstractions for relational database interactions, schema mapping, and query building.

### Q17: What is a foreign key?
A foreign key is a database column or set of columns that establishes a relational link between rows in one table and a primary key row in another table, enforcing referential integrity.

### Q18: What is a database transaction?
A database transaction is a sequence of SQL operations executed as a single atomic unit of work following ACID properties (Atomicity, Consistency, Isolation, Durability). All operations commit together, or rollback completely on error.

### Q19: Why do we use Alembic?
Alembic manages database schema migrations, allowing developers to generate, track, apply, and rollback database schema changes chronologically in sync with SQLAlchemy Python models.

### Q20: What is JWT?
JWT (JSON Web Token) is a compact, URL-safe standard (RFC 7519) for transmitting signed JSON claims between client and server for stateless authentication.

### Q21: What is the difference between Authentication and Authorization?
- **Authentication**: Verifying **who you are** (e.g., checking email/password or JWT token).
- **Authorization**: Verifying **what you are allowed to do** (e.g., checking user roles or resource ownership).

### Q22: Why should passwords be hashed?
Passwords must be hashed using one-way salted hashing algorithms (like bcrypt) so that even if the database is compromised, cleartext passwords are never exposed to attackers.

### Q23: What is Dependency Injection in FastAPI?
Dependency Injection (`Depends()`) is a design pattern where FastAPI handles instantiating and injecting required resources (e.g., database sessions, current user authentication, configuration settings) into route handlers automatically.

### Q24: What is Middleware?
Middleware is software that hooks into the ASGI request-response lifecycle, processing requests before they reach route handlers and modifying responses before they are returned to clients (e.g., adding headers, logging, CORS).

### Q25: Why do we write automated tests?
Automated tests (using Pytest) prove that software works correctly, prevent regressions when refactoring code, document system requirements, and enable continuous integration and deployment.

---

## 🎯 PART 4 — Interview Round Q&A Reference

### Beginner Level
- **What is Python?**: Python is an interpreted, high-level, dynamically-typed, general-purpose programming language known for readability.
- **Why is Python dynamically typed?**: Variable types are bound at runtime during evaluation rather than compile time.
- **Set vs Dictionary?**: Sets store unique elements; dictionaries store key-value pairs. Both use hashtables for $O(1)$ operations.

### Intermediate Level
- **What is a Generator?**: A lazy evaluation function using `yield` to stream values on demand, saving memory.
- **What is OOP?**: A paradigm structuring software around data objects containing attributes and methods.
- **What is Dependency Injection?**: Decoupling components by injecting dependencies externally rather than hardcoding their creation inside functions.

### Backend Level
- **What happens when you call an API?**: Client sends HTTP request → DNS resolves host → TCP connection established → Server ASGI handler routes request → FastAPI validates Pydantic model → Business logic executes DB transaction → JSON response returned to client.
- **Why is JWT stateless?**: The token contains digitally signed claims inside its payload, allowing the backend to authenticate requests without storing active session IDs in a central database or memory cache.

---

## 📊 PART 5 — Self-Assessment Scorecard (12 Domains)

| Domain | Rating (/10) | Status |
| :--- | :---: | :--- |
| **Python Fundamentals** | 10/10 | Mastered 🔥 |
| **Data Structures** | 10/10 | Mastered 🔥 |
| **Functions & Scope** | 10/10 | Mastered 🔥 |
| **OOP & Design Principles** | 10/10 | Mastered 🔥 |
| **Advanced Python (Generators/Decos/CM)** | 10/10 | Mastered 🔥 |
| **FastAPI Framework** | 10/10 | Mastered 🔥 |
| **SQL & Relational Databases** | 10/10 | Mastered 🔥 |
| **SQLAlchemy 2.0 ORM** | 10/10 | Mastered 🔥 |
| **Alembic Database Migrations** | 10/10 | Mastered 🔥 |
| **JWT Auth & Password Hashing** | 10/10 | Mastered 🔥 |
| **Pytest Automated Testing** | 10/10 | Mastered 🔥 |
| **Backend Clean Architecture & Logging** | 10/10 | Mastered 🔥 |
| **TOTAL SCORE** | **120 / 120** | **EXCELLENT (Grade: A+)** |
