# ==============================================================================
# Program    : Health Check & Observability Endpoint Tests (test_health.py)
# Objective  : Test GET /health (Liveness) and GET /health/ready (Readiness).
# Concept    : Production Observability & Health Probing
# Why Used   : Validates liveness and readiness probe HTTP contracts for load balancers.
# ==============================================================================

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_liveness_probe_returns_200(client):
    """Test GET /health returns HTTP 200 OK and status='alive'."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"
    assert "app_name" in data
    assert "environment" in data

def test_readiness_probe_success_returns_200(client):
    """Test GET /health/ready returns HTTP 200 OK and database='connected' when database is reachable."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["database"] == "connected"

def test_readiness_probe_database_failure_returns_503(client):
    """Test GET /health/ready returns HTTP 503 Service Unavailable when database connection fails."""
    with patch("app.routers.health.check_database_readiness", return_value=False):
        response = client.get("/health/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["database"] == "disconnected"

def test_root_index_endpoint_returns_documentation_urls(client):
    """Test GET / returns API index and documentation sitemap."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["documentation"] == "/docs"
    assert data["redoc"] == "/redoc"
