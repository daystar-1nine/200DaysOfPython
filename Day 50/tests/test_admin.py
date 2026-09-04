"""
===============================================================================
DAY 50 — ADMIN ROLE-BASED ACCESS CONTROL (RBAC) TESTS
===============================================================================
This module tests administration endpoints (/admin/users, /admin/tasks).
===============================================================================
"""

from typing import Dict
from fastapi.testclient import TestClient


def test_admin_get_users_allowed(client: TestClient, admin_auth_headers: Dict[str, str]) -> None:
    """Test admin user accessing GET /admin/users succeeds with 200 OK."""
    response = client.get("/admin/users", headers=admin_auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1


def test_normal_user_get_admin_users_forbidden(client: TestClient, user_auth_headers: Dict[str, str]) -> None:
    """Test normal user accessing GET /admin/users returns 403 Forbidden."""
    response = client.get("/admin/users", headers=user_auth_headers)
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "FORBIDDEN"


def test_admin_get_tasks_allowed(
    client: TestClient,
    user_auth_headers: Dict[str, str],
    admin_auth_headers: Dict[str, str],
) -> None:
    """Test admin user accessing GET /admin/tasks succeeds and sees all user tasks."""
    client.post("/tasks", json={"title": "Normal User Task"}, headers=user_auth_headers)

    response = client.get("/admin/tasks", headers=admin_auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert data[0]["title"] == "Normal User Task"


def test_normal_user_get_admin_tasks_forbidden(client: TestClient, user_auth_headers: Dict[str, str]) -> None:
    """Test normal user accessing GET /admin/tasks returns 403 Forbidden."""
    response = client.get("/admin/tasks", headers=user_auth_headers)
    assert response.status_code == 403
