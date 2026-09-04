"""
===============================================================================
DAY 50 — HEALTH & OBSERVABILITY PROBES TESTS
===============================================================================
This module tests Liveness (GET /health) and Database Readiness (GET /health/ready).
===============================================================================
"""

from fastapi.testclient import TestClient


def test_health_liveness_endpoint(client: TestClient) -> None:
    """Test GET /health returns status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "TaskFlow API"


def test_health_readiness_endpoint(client: TestClient) -> None:
    """Test GET /health/ready verifies database connectivity."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["database"] == "connected"
