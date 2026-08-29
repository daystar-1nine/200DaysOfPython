# 🐍 Day 42/200 – Masterclass Notes: REST APIs with FastAPI

🎯 **Goal:** Shift from API consumer to API creator—mastering the FastAPI web framework, Uvicorn ASGI web server, OpenAPI auto-generated interactive documentation (`/docs` & `/redoc`), path vs query parameters, Pydantic `BaseModel` request/response data validation, HTTP CRUD methods (GET, POST, PUT, PATCH, DELETE), `HTTPException` error responses, pagination, keyword searching, and API endpoint unit testing with `fastapi.testclient.TestClient`.

---

## 📌 Executive Summary & Key Takeaways

- **FastAPI Framework Architecture:**
  - Built on **Starlette** (for ASGI web routing) and **Pydantic** (for data validation and serialization).
  - Uses standard Python type hints to generate OpenAPI schema specifications automatically at `/docs` (Swagger UI) and `/redoc` (ReDoc).
- **Path Parameters vs Query Parameters:**
  - **Path Parameters (`/users/{user_id}`):** Identifies a specific resource by unique identifier (e.g. `@app.get("/users/{user_id}") def get_user(user_id: int)`).
  - **Query Parameters (`/users?skip=0&limit=10`):** Optional or scalar parameters declared as function arguments not present in the URL path string (e.g. `def list_users(skip: int = 0, limit: int = 10)`).
- **Pydantic Validation & Fields:**
  - `BaseModel`: Base class for request/response schemas.
  - `Field(min_length=2, gt=0, lt=120)`: Enforces validation rules on string lengths and numerical bounds.
- **PUT vs PATCH Semantics:**
  - `PUT /users/{id}`: Replaces the complete resource data object.
  - `PATCH /users/{id}`: Partially modifies specified attributes using optional fields (`name: str | None = None`).
- **HTTP Status Codes in FastAPI:**
  - `raise HTTPException(status_code=404, detail="User not found")` produces structured JSON error responses with standard HTTP status codes.

---

## 📖 Topic 1: FastAPI Endpoint & Pydantic Validation

```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, EmailStr

app = FastAPI(title="User Management API")

class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=50, description="Full Name")
    email: str = Field(..., description="User Email Address")
    age: int = Field(..., gt=0, lt=120, description="User Age in Years")

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    age: int

@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(payload: UserCreate):
    # FastAPI automatically validates payload against UserCreate schema
    new_user = {"id": 1, **payload.model_dump()}
    return new_user
```

---

## ⚡ Master Cheat Sheet

```python
# FastAPI & Pydantic Cheat Sheet

# 1. Start Uvicorn Server
# uvicorn app.main:app --reload --port 8000

# 2. Path Parameter with Type Conversion
@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id}

# 3. Query Parameters with Defaults
@app.get("/users")
def list_users(skip: int = 0, limit: int = 10, search: str | None = None):
    return {"skip": skip, "limit": limit, "search": search}

# 4. HTTP Exception
@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id > 100:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id}
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Putting Business Logic inside Route Handlers:**
   - ❌ Writing in-memory list mutations or database queries directly inside `@app.get` or `@app.post` route functions.
   - ✅ Delegate data access and mutations to a dedicated `UserService` layer.

2. **Returning Unfiltered Request Models:**
   - ❌ Returning sensitive fields (e.g., password hashes) by omitting `response_model=UserResponse`.
   - ✅ Always specify `response_model` on endpoints to filter and document output JSON schemas strictly.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: Why does FastAPI generate automatic documentation at `/docs`?
**Answer:** FastAPI inspects route type annotations and Pydantic schema models to generate a standard OpenAPI JSON specification document. Interactive tools like Swagger UI (`/docs`) and ReDoc (`/redoc`) consume this JSON spec to render interactive API testing consoles automatically.

### Q2: How does Pydantic validation differ between request models and dict parameters?
**Answer:** Passing `payload: dict` accepts any arbitrary JSON object without type or constraint enforcement. Defining `payload: UserCreate` validates data types, string bounds, and numerical constraints automatically before the function executes, raising HTTP 422 Unprocessable Content if validation fails.

---

## 📝 Recap Checklist
- [x] Understood Client-Server HTTP API creation with FastAPI and Uvicorn.
- [x] Built FastAPI application routes (`GET`, `POST`, `PUT`, `PATCH`, `DELETE`).
- [x] Applied Pydantic `BaseModel` & `Field` validation constraints.
- [x] Created `UserService` layer for in-memory user CRUD operations, pagination, and keyword search.
- [x] Designed Pytest test suite using `TestClient` for 15+ test cases.
