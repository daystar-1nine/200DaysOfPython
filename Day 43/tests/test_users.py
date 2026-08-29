# ==============================================================================
# Test Suite : User API Endpoints Tests (test_users.py)
# Objective  : Integration testing of REST API V2 endpoints using FastAPI TestClient.
# Concept    : API Integration Testing
# Why Used   : Asserts end-to-end HTTP request execution through routers and services.
# ==============================================================================

import os
import sys
from fastapi.testclient import TestClient

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.main import app

client = TestClient(app)

def test_api_read_root():
    res = client.get("/")
    assert res.status_code == 200
    assert "User Management API V2" in res.json()["message"]

def test_api_read_config():
    res = client.get("/config")
    assert res.status_code == 200
    assert res.json()["environment"] == "development"

def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "healthy"}

def test_api_about():
    res = client.get("/about")
    assert res.status_code == 200
    assert res.json()["version"] == "2.0.0"

def test_api_get_profile():
    res = client.get("/profile")
    assert res.status_code == 200
    profile = res.json()
    assert profile["name"] == "Suraj Sawant"
    assert profile["role"] == "user"

def test_api_list_users():
    res = client.get("/users")
    assert res.status_code == 200
    assert len(res.json()) >= 4

def test_api_list_users_pagination():
    res = client.get("/users?skip=1&limit=2")
    assert res.status_code == 200
    assert len(res.json()) == 2

def test_api_search_users():
    res = client.get("/users/search?name=suraj")
    assert res.status_code == 200
    matches = res.json()
    assert len(matches) >= 1
    assert matches[0]["name"] == "Suraj Sawant"

def test_api_get_user_valid():
    res = client.get("/users/1")
    assert res.status_code == 200
    assert res.json()["id"] == 1

def test_api_get_user_not_found():
    res = client.get("/users/999")
    assert res.status_code == 404

def test_api_create_user():
    payload = {"name": "Test User", "email": "testuser@example.com", "age": 28}
    res = client.post("/users", json=payload)
    assert res.status_code == 201
    assert res.json()["email"] == "testuser@example.com"

def test_api_create_user_duplicate():
    payload = {"name": "Duplicate User", "email": "suraj@example.com", "age": 22}
    res = client.post("/users", json=payload)
    assert res.status_code == 409

def test_api_replace_user():
    payload = {"name": "Suraj Replaced", "email": "suraj.rep@example.com", "age": 22}
    res = client.put("/users/1", json=payload)
    assert res.status_code == 200
    assert res.json()["name"] == "Suraj Replaced"

def test_api_patch_user():
    payload = {"age": 24}
    res = client.patch("/users/1", json=payload)
    assert res.status_code == 200
    assert res.json()["age"] == 24

def test_api_delete_user():
    res = client.delete("/users/4")
    assert res.status_code == 200
    assert client.get("/users/4").status_code == 404

def test_api_products_router():
    res = client.get("/products")
    assert res.status_code == 200
    assert len(res.json()) >= 2
