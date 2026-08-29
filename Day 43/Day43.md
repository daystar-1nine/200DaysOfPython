# 🐍 Day 43/200 – Masterclass Notes: FastAPI Architecture & Dependency Injection

🎯 **Goal:** Transition from basic script-based API routes to clean, production-grade **Layered Backend Architecture**—mastering `APIRouter`, FastAPI's `Depends()` Dependency Injection engine, Service Layer, Repository Pattern, authentication dependencies, configuration injection, and `app.dependency_overrides` for testing.

---

## 📌 Executive Summary & Key Takeaways

- **The Layered Architecture Request Pipeline:**
  ```text
  Client ──> Router (HTTP) ──> Dependencies (Auth/Config) ──> Service (Business Logic) ──> Repository (Data Access) ──> Storage
  ```
- **FastAPI `APIRouter`:**
  - Modularizes endpoints into distinct router domain files (`routers/users.py`, `routers/products.py`).
  - Combined in `main.py` using `app.include_router(users_router)`.
- **Dependency Injection (`from fastapi import Depends`):**
  - **Concept:** Inversion of Control (IoC). Instead of instantiating dependencies (database sessions, current user, configuration) inside every endpoint, FastAPI resolves dependencies and injects them as function arguments.
  - **Syntax:** `user: dict = Depends(get_current_user)` or `service: UserService = Depends(get_user_service)`.
- **Service vs Repository Layer:**
  - **Repository Layer (`UserRepository`):** Performs data access operations (`get_all`, `get_by_id`, `create`, `update`, `delete`) directly on storage without containing business validation rules.
  - **Service Layer (`UserService`):** Orchestrates business validation (checking unique emails, enforcing minimum age rules, calculating metrics) and delegates persistence calls to the repository.
- **Dependency Overrides in Testing (`app.dependency_overrides`):**
  - Allows swapping production dependencies (e.g. real database connections or real external auth services) with mock or test fixtures during unit testing (`app.dependency_overrides[get_current_user] = mock_get_current_user`).

---

## 📖 Topic 1: Dependency Injection & Layered Providers

```python
from fastapi import FastAPI, Depends, APIRouter
from pydantic import BaseModel

app = FastAPI()
router = APIRouter(prefix="/users", tags=["Users"])

# 1. Dependency Factory
def get_user_repository():
    return UserRepository()

def get_user_service(repo=Depends(get_user_repository)):
    return UserService(repo)

# 2. Router Endpoint with Injected Service
@router.get("/{user_id}")
def get_user(user_id: int, service: UserService = Depends(get_user_service)):
    return service.get_user(user_id)

app.include_router(router)
```

---

## ❓ Concept Challenge (Self-Assessment Answers)

1. **What is `APIRouter`?**
   - An `APIRouter` is a modular web router in FastAPI that allows grouping endpoints into separate domain modules with common path prefixes and tags.
2. **What does `Depends()` do?**
   - `Depends()` signals to FastAPI that a function parameter requires a dependency to be executed and injected automatically before calling the route handler.
3. **What is dependency injection?**
   - A design pattern where an object receives its dependencies from an external framework rather than creating them internally, increasing testability and decoupling.
4. **What belongs in a Router?**
   - HTTP request parsing, status codes, route paths, query/path parameter declarations, and delegating calls to injected services.
5. **What belongs in a Service?**
   - Business validation rules, domain logic, calculations, and orchestrating workflow operations.
6. **What belongs in a Repository?**
   - Direct data storage access, querying, filtering, and persistence CRUD operations.
7. **Why shouldn't the router contain all business logic?**
   - Keeping business logic inside routers creates bloated, untestable code that cannot be reused outside HTTP requests.
8. **Why are dependency overrides useful for testing?**
   - `app.dependency_overrides` lets unit tests inject mock data sources or mock users without altering production code.

---

## ⚡ Master Cheat Sheet

```python
# FastAPI Architecture & Dependency Injection Cheat Sheet

# 1. Register Router with Prefix and Tags
app.include_router(users_router, prefix="/api/v1")

# 2. Injected Dependency Function
def get_current_user(token: str = Header(...)):
    return {"id": 1, "username": "suraj"}

# 3. Use Dependency in Endpoint
@app.get("/profile")
def get_profile(current_user: dict = Depends(get_current_user)):
    return current_user

# 4. Override Dependency for Unit Tests
app.dependency_overrides[get_current_user] = lambda: {"id": 99, "username": "test_user"}
```

---

## 📝 Recap Checklist
- [x] Mastered modular routing with `APIRouter`.
- [x] Understood FastAPI Dependency Injection via `Depends()`.
- [x] Implemented Repository Pattern (`UserRepository`, `ProductRepository`) and Service Layer (`UserService`, `ProductService`).
- [x] Created `get_current_user` fake authentication dependency.
- [x] Applied `app.dependency_overrides` for testing isolation.
- [x] Built independent practice exercises (Exercises 1–4).
- [x] Built 20+ Pytest tests covering all layers.
