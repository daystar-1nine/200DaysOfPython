# ==============================================================================
# Test Suite : Dependency Overrides Unit Tests (test_overrides.py)
# Objective  : Demonstrate FastAPI app.dependency_overrides for mock testing dependencies (Bonus Challenge).
# Concept    : FastAPI Dependency Overrides (Day 43 requirement)
# Why Used   : Swaps production dependencies with test doubles cleanly during unit tests.
# ==============================================================================

import os
import sys
from fastapi.testclient import TestClient

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.config import get_settings
from app.dependencies.auth import get_current_user
from app.main import app

client = TestClient(app)

def test_override_auth_dependency():
    def mock_get_current_user():
        return {
            "id": 999,
            "name": "Mock Test User",
            "email": "mock@example.com",
            "role": "superuser"
        }

    # What is used : app.dependency_overrides
    # Why it is used: Overrides get_current_user dependency during test execution
    app.dependency_overrides[get_current_user] = mock_get_current_user

    res = client.get("/profile")
    assert res.status_code == 200
    profile = res.json()
    assert profile["id"] == 999
    assert profile["name"] == "Mock Test User"
    assert profile["role"] == "superuser"

    # Reset overrides after test
    app.dependency_overrides.clear()

def test_override_settings_dependency():
    def mock_get_settings():
        return {"app_title": "Test Suite App", "environment": "test"}

    app.dependency_overrides[get_settings] = mock_get_settings

    res = client.get("/config")
    assert res.status_code == 200
    assert res.json()["environment"] == "test"

    app.dependency_overrides.clear()
