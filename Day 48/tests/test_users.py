# ==============================================================================
# Program    : User Profile & Admin Authorization Tests (test_users.py)
# Objective  : Test /users/me, /users/me/orders, /admin/users (403 for user, 200 for admin), and cascading deletions.
# Concept    : Authorization Scoping & RBAC Verification
# Why Used   : Ensures user privacy and enforces administrator permission boundaries.
# ==============================================================================

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_get_current_user_profile_success(client, normal_user, normal_user_headers):
    """Test retrieving authenticated profile returns user data excluding password_hash."""
    response = client.get("/users/me", headers=normal_user_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == normal_user.id
    assert data["email"] == normal_user.email
    assert data["role"] == "user"
    assert "password_hash" not in data

def test_get_current_user_orders_empty(client, normal_user_headers):
    """Test retrieving authenticated user's orders returns empty list when no orders exist."""
    response = client.get("/users/me/orders", headers=normal_user_headers)
    assert response.status_code == 200
    assert response.json() == []

def test_admin_list_users_as_normal_user_returns_403(client, normal_user_headers):
    """Test regular user attempting to call /admin/users gets HTTP 403 Forbidden."""
    response = client.get("/admin/users", headers=normal_user_headers)
    assert response.status_code == 403
    assert "privileges are required" in response.json()["detail"].lower()

def test_admin_list_users_as_admin_user_success(client, admin_user, admin_user_headers):
    """Test admin calling /admin/users gets HTTP 200 OK and user collection."""
    response = client.get("/admin/users", headers=admin_user_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    emails = [u["email"] for u in data]
    assert admin_user.email in emails

def test_admin_list_users_unauthenticated_returns_401(client):
    """Test calling /admin/users without authentication header returns HTTP 401 Unauthorized."""
    response = client.get("/admin/users")
    assert response.status_code == 401

def test_health_check_endpoint(client):
    """Test health check root endpoint returns 200 OK and status metadata."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
