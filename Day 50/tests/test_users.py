"""
===============================================================================
DAY 50 — USER PROFILE INTEGRATION TESTS
===============================================================================
This module tests user profile endpoint access (/users/me).
===============================================================================
"""

from typing import Dict
from fastapi.testclient import TestClient
from app.models.user import User


def test_get_current_user_profile_success(client: TestClient, normal_user: User, user_auth_headers: Dict[str, str]) -> None:
    """Test retrieving currently authenticated user profile."""
    response = client.get("/users/me", headers=user_auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == normal_user.id
    assert data["email"] == normal_user.email
    assert data["name"] == normal_user.name
    assert data["role"] == "user"


def test_get_current_user_unauthorized_fails(client: TestClient) -> None:
    """Test retrieving user profile without auth headers fails with 401."""
    response = client.get("/users/me")
    assert response.status_code == 401
