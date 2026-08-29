# ==============================================================================
# Test Suite : User Management API Endpoint Tests (test_users.py)
# Objective  : Test FastAPI endpoints (GET, POST, PUT, PATCH, DELETE, /search, /health, /about) using TestClient.
# Concept    : API Endpoint Testing with FastAPI TestClient
# Why Used   : Asserts HTTP status codes and JSON response bodies without running live Uvicorn server.
# ==============================================================================

import os
import sys
import pytest
from fastapi.testclient import TestClient

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_about_endpoint():
    response = client.get("/about")
    assert response.status_code == 200
    assert response.json()["framework"] == "FastAPI"

def test_list_users_initial():
    response = client.get("/users")
    assert response.status_code == 200
    users = response.json()
    assert len(users) >= 4
    assert users[0]["name"] == "Suraj Sawant"

def test_list_users_pagination():
    response = client.get("/users?skip=1&limit=2")
    assert response.status_code == 200
    users = response.json()
    assert len(users) == 2
    assert users[0]["id"] == 2

def test_search_users():
    response = client.get("/users/search?name=suraj")
    assert response.status_code == 200
    matches = response.json()
    assert len(matches) == 1
    assert matches[0]["name"] == "Suraj Sawant"

def test_get_user_valid():
    response = client.get("/users/1")
    assert response.status_code == 200
    user = response.json()
    assert user["id"] == 1
    assert user["email"] == "suraj@example.com"

def test_get_user_not_found():
    response = client.get("/users/999")
    assert response.status_code == 404
    assert "was not found" in response.json()["detail"]

def test_get_user_invalid_path_param_type():
    response = client.get("/users/abc")
    assert response.status_code == 422

def test_create_user_success():
    payload = {
        "name": "New Developer",
        "email": "newdev@example.com",
        "age": 24
    }
    response = client.post("/users", json=payload)
    assert response.status_code == 201
    created = response.json()
    assert created["id"] == 5
    assert created["name"] == "New Developer"

def test_create_user_duplicate_email_conflict():
    payload = {
        "name": "Duplicate User",
        "email": "suraj@example.com",
        "age": 22
    }
    response = client.post("/users", json=payload)
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]

def test_replace_user_put():
    payload = {
        "name": "Suraj Sawant Updated",
        "email": "suraj.updated@example.com",
        "age": 22
    }
    response = client.put("/users/1", json=payload)
    assert response.status_code == 200
    updated = response.json()
    assert updated["name"] == "Suraj Sawant Updated"

def test_patch_user():
    payload = {"age": 23}
    response = client.patch("/users/1", json=payload)
    assert response.status_code == 200
    patched = response.json()
    assert patched["age"] == 23

def test_delete_user():
    response = client.delete("/users/4")
    assert response.status_code == 200
    assert response.json()["id"] == 4

    # Verify 404 after deletion
    get_resp = client.get("/users/4")
    assert get_resp.status_code == 404
