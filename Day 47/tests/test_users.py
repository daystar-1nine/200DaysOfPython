# ==============================================================================
# Test Suite : User Profile & Admin Authorization Tests (test_users.py)
# Objective  : Test /users/me, /users/me/orders, and /admin/users role authorization.
# Concept    : Protected Routes, 401 Unauthorized, and 403 Forbidden Verification
# Why Used   : Ensures normal users are barred from admin endpoints (403) and unauthenticated requests fail (401).
# ==============================================================================

import os
import sys
import pytest

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

def test_read_current_user_me_success(client, test_user, user_token_headers):
    response = client.get("/users/me", headers=user_token_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == test_user.id
    assert data["email"] == test_user.email
    assert data["role"] == "user"

def test_read_current_user_me_unauthenticated_fails(client):
    response = client.get("/users/me")
    assert response.status_code == 401

def test_admin_list_users_success_for_admin(client, admin_token_headers):
    response = client.get("/admin/users", headers=admin_token_headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_admin_list_users_forbidden_for_regular_user(client, user_token_headers):
    response = client.get("/admin/users", headers=user_token_headers)
    assert response.status_code == 403
    assert "Administrator privileges are required" in response.json()["detail"]

def test_admin_list_users_unauthorized_for_missing_token(client):
    response = client.get("/admin/users")
    assert response.status_code == 401

def test_read_current_user_me_orders_returns_empty_list(client, user_token_headers):
    response = client.get("/users/me/orders", headers=user_token_headers)
    assert response.status_code == 200
    assert response.json() == []
