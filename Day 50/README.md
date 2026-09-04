# 🚀 TaskFlow API — Production-Style Task Management REST Backend (Day 50 Milestone)

> **TaskFlow API** is a production-grade, asynchronous RESTful backend application built with **FastAPI**, **SQLAlchemy 2.0 ORM**, **Alembic**, **PyJWT**, **Passlib bcrypt**, and **Pytest**.

---

## 📌 Architecture Overview

TaskFlow API follows a strict **Clean Layered Architecture**:

```text
Client Request
      │
      ↓
ASGI Middleware (X-Request-ID & Process-Time-Ms)
      │
      ↓
FastAPI APIRouter (/auth, /users, /tasks, /admin, /health)
      │
      ↓
OAuth2 Bearer JWT Authentication & RBAC Authorization
      │
      ↓
Pydantic Request Validation Schemas
      │
      ↓
Service Layer (AuthService, UserService, TaskService)
      │
      ↓
Repository Layer (UserRepository, TaskRepository)
      │
      ↓
SQLAlchemy 2.0 ORM (User, Task)
      │
      ↓
Database Engine (PostgreSQL / SQLite)
```

---

## 🛠️ API Endpoint Reference

### 🔐 Authentication (`/auth`)
| Method | Endpoint | Description | Auth Required | Status Code |
| :--- | :--- | :--- | :---: | :---: |
| `POST` | `/auth/register` | Register a new user account | ❌ | `201 Created` |
| `POST` | `/auth/login` | Authenticate credentials & issue JWT access token | ❌ | `200 OK` |

### 👤 User Operations (`/users`)
| Method | Endpoint | Description | Auth Required | Status Code |
| :--- | :--- | :--- | :---: | :---: |
| `GET` | `/users/me` | Retrieve current authenticated user profile | ✅ Bearer | `200 OK` |

### 📋 Task Management (`/tasks`)
| Method | Endpoint | Description | Auth Required | Status Code |
| :--- | :--- | :--- | :---: | :---: |
| `POST` | `/tasks` | Create a new task item | ✅ Bearer | `201 Created` |
| `GET` | `/tasks` | List user tasks (`status`, `priority`, `search`, pagination) | ✅ Bearer | `200 OK` |
| `GET` | `/tasks/{id}` | Get single task detail by ID (User isolated) | ✅ Bearer | `200 OK` |
| `PUT` | `/tasks/{id}` | Full task entity replacement | ✅ Bearer | `200 OK` |
| `PATCH` | `/tasks/{id}` | Partial task attribute modification | ✅ Bearer | `200 OK` |
| `DELETE` | `/tasks/{id}` | Delete task entity | ✅ Bearer | `204 No Content` |

### 👑 Admin Operations (`/admin`)
| Method | Endpoint | Description | Auth Required | Status Code |
| :--- | :--- | :--- | :---: | :---: |
| `GET` | `/admin/users` | List all registered users across system | 👑 Admin | `200 OK` |
| `GET` | `/admin/tasks` | List all tasks across all users | 👑 Admin | `200 OK` |

### 🩺 Health & Observability
| Method | Endpoint | Description | Auth Required | Status Code |
| :--- | :--- | :--- | :---: | :---: |
| `GET` | `/health` | Application process liveness probe | ❌ | `200 OK` |
| `GET` | `/health/ready` | Database readiness probe (`SELECT 1`) | ❌ | `200 OK` |

---

## 🧪 Running Automated Pytest Suite

```bash
# Execute complete automated Pytest test suite (45+ test cases)
pytest "Day 50/tests/" -v
```
