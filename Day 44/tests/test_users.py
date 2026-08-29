# ==============================================================================
# Test Suite : User REST API Database Integration Tests (test_users.py)
# Objective  : Integration testing of database-backed REST API endpoints using TestClient.
# Concept    : Integration Testing with Dependency Overrides for In-Memory Database Session
# Why Used   : Asserts end-to-end HTTP request processing against actual database tables.
# ==============================================================================

import os
import sys
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import Base, get_db
from app.main import app

# Setup isolated in-memory test database with StaticPool to share connection across threads
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Apply dependency override for tests
app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

def test_api_read_root():
    res = client.get("/")
    assert res.status_code == 200
    assert "User Management API V3" in res.json()["message"]

def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_api_about():
    res = client.get("/about")
    assert res.status_code == 200
    assert res.json()["version"] == "3.0.0"

def test_api_create_user_valid():
    payload = {"name": "Suraj Sawant", "email": "suraj@example.com", "age": 21}
    res = client.post("/users", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["id"] == 1
    assert data["email"] == "suraj@example.com"

def test_api_create_user_duplicate_email():
    payload = {"name": "Suraj Sawant", "email": "suraj@example.com", "age": 21}
    client.post("/users", json=payload)
    res = client.post("/users", json=payload)
    assert res.status_code == 409

def test_api_create_user_invalid_age():
    payload = {"name": "Suraj Sawant", "email": "suraj@example.com", "age": -5}
    res = client.post("/users", json=payload)
    assert res.status_code == 422

def test_api_create_user_invalid_name():
    payload = {"name": "S", "email": "suraj@example.com", "age": 21}
    res = client.post("/users", json=payload)
    assert res.status_code == 422

def test_api_get_users_list():
    client.post("/users", json={"name": "Suraj", "email": "suraj@example.com", "age": 21})
    client.post("/users", json={"name": "Alex", "email": "alex@example.com", "age": 25})
    res = client.get("/users")
    assert res.status_code == 200
    assert len(res.json()) == 2

def test_api_get_users_pagination():
    for i in range(1, 6):
        client.post("/users", json={"name": f"User{i}", "email": f"u{i}@example.com", "age": 20 + i})
    res = client.get("/users?skip=1&limit=2")
    assert res.status_code == 200
    assert len(res.json()) == 2

def test_api_search_users_db_query():
    client.post("/users", json={"name": "Suraj Sawant", "email": "suraj@example.com", "age": 21})
    client.post("/users", json={"name": "Suraj Kumar", "email": "suraj.k@example.com", "age": 24})
    client.post("/users", json={"name": "Jane Smith", "email": "jane@example.com", "age": 28})

    res = client.get("/users/search?name=suraj")
    assert res.status_code == 200
    matches = res.json()
    assert len(matches) == 2

def test_api_get_user_valid():
    res_c = client.post("/users", json={"name": "Suraj", "email": "suraj@example.com", "age": 21})
    user_id = res_c.json()["id"]
    res = client.get(f"/users/{user_id}")
    assert res.status_code == 200
    assert res.json()["name"] == "Suraj"

def test_api_get_user_not_found():
    res = client.get("/users/999")
    assert res.status_code == 404

def test_api_replace_user_put():
    res_c = client.post("/users", json={"name": "Suraj", "email": "suraj@example.com", "age": 21})
    user_id = res_c.json()["id"]
    payload = {"name": "Suraj Replaced", "email": "suraj.rep@example.com", "age": 22}
    res = client.put(f"/users/{user_id}", json=payload)
    assert res.status_code == 200
    assert res.json()["name"] == "Suraj Replaced"

def test_api_replace_user_not_found():
    payload = {"name": "Suraj Replaced", "email": "suraj.rep@example.com", "age": 22}
    res = client.put("/users/999", json=payload)
    assert res.status_code == 404

def test_api_patch_user_valid():
    res_c = client.post("/users", json={"name": "Suraj", "email": "suraj@example.com", "age": 21})
    user_id = res_c.json()["id"]
    payload = {"age": 25}
    res = client.patch(f"/users/{user_id}", json=payload)
    assert res.status_code == 200
    assert res.json()["age"] == 25

def test_api_delete_user_valid():
    res_c = client.post("/users", json={"name": "ToDelete", "email": "del@example.com", "age": 20})
    user_id = res_c.json()["id"]
    res = client.delete(f"/users/{user_id}")
    assert res.status_code == 200
    assert client.get(f"/users/{user_id}").status_code == 404

def test_api_delete_user_not_found():
    res = client.delete("/users/999")
    assert res.status_code == 404
