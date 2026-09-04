"""
===============================================================================
DAY 50 — STANDARDIZED GLOBAL ERROR HANDLING TESTS
===============================================================================
This module tests global exception handler conversion to standardized error JSON.
===============================================================================
"""

from typing import Dict
from fastapi.testclient import TestClient


def test_custom_exception_standardized_json_response(client: TestClient, user_auth_headers: Dict[str, str]) -> None:
    """Test custom TaskFlowException (NotFoundError) formatted as standardized error payload."""
    response = client.get("/tasks/99999", headers=user_auth_headers)  # Non-existent task returns 404 NotFoundError
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "NOT_FOUND"
    assert "Task with ID 99999 not found" in data["error"]["message"]
    assert "request_id" in data["error"]


def test_validation_error_standardized_json_response(client: TestClient) -> None:
    """Test Pydantic validation error formatted as 422 error payload."""
    invalid_payload = {"name": "A"}  # Short name (< 2 chars) and missing email/password
    response = client.post("/auth/register", json=invalid_payload)
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "UNPROCESSABLE_ENTITY"
    assert "details" in data["error"]
