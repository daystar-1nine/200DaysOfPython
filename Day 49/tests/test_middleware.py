# ==============================================================================
# Program    : Middleware Unit & Integration Tests (test_middleware.py)
# Objective  : Test X-Request-ID propagation, client header preservation, and Process-Time-Ms latency header.
# Concept    : ASGI Middleware Testing
# Why Used   : Verifies request tracing and execution latency headers across all endpoints.
# ==============================================================================

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_middleware_generates_x_request_id_header(client):
    """Test middleware automatically generates X-Request-ID response header if omitted by client."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) >= 8

def test_middleware_preserves_client_supplied_x_request_id_header(client):
    """Test middleware preserves custom X-Request-ID passed in client HTTP request headers."""
    custom_request_id = "test_custom_req_id_12345"
    headers = {"X-Request-ID": custom_request_id}
    response = client.get("/health", headers=headers)

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == custom_request_id

def test_middleware_attaches_process_time_ms_header(client):
    """Test timing middleware attaches Process-Time-Ms execution duration header to response."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "Process-Time-Ms" in response.headers
    duration_str = response.headers["Process-Time-Ms"]
    assert float(duration_str) >= 0.0

def test_middleware_propagates_request_id_to_error_response(client):
    """Test X-Request-ID is present in standardized error JSON payloads and headers."""
    custom_id = "err_req_9999"
    headers = {"X-Request-ID": custom_id}
    response = client.get("/products/99999", headers=headers)

    assert response.status_code == 404
    assert response.headers["X-Request-ID"] == custom_id
    data = response.json()
    assert data["error"]["request_id"] == custom_id
