"""
===============================================================================
DAY 50 — MIDDLEWARE INTEGRATION TESTS
===============================================================================
This module tests X-Request-ID and Process-Time-Ms response headers.
===============================================================================
"""

from fastapi.testclient import TestClient


def test_request_id_middleware_header_inserted(client: TestClient) -> None:
    """Test X-Request-ID header is present in HTTP response."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0


def test_process_time_ms_middleware_header_inserted(client: TestClient) -> None:
    """Test Process-Time-Ms latency header is present in HTTP response."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "Process-Time-Ms" in response.headers
    assert float(response.headers["Process-Time-Ms"]) >= 0.0
