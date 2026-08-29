# ==============================================================================
# Test Suite : API Client HTTP Unit Tests (test_client.py)
# Objective  : Test GET, POST, PUT, PATCH, DELETE operations, error exceptions, and SDK resources using HTTP mocking.
# Concept    : Unit Testing HTTP Network Clients with unittest.mock
# Why Used   : Prevents real network requests, rate limiting, and test flakiness during automated builds.
# ==============================================================================

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import pytest
import requests

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from api_explorer.client import APIClient
from api_explorer.exceptions import (
    APIError,
    APIConnectionError,
    APINotFoundError,
    APITimeoutError
)

@patch("requests.Session.request")
def test_client_get_request(mock_request):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"id": 1, "name": "Suraj"}]
    mock_request.return_value = mock_resp

    client = APIClient("https://api.example.com")
    res = client.get("/users")

    assert len(res) == 1
    assert res[0]["name"] == "Suraj"
    mock_request.assert_called_once_with(
        method="GET",
        url="https://api.example.com/users",
        params=None,
        json=None,
        timeout=10.0
    )

@patch("requests.Session.request")
def test_client_post_request(mock_request):
    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {"id": 101, "title": "New Post"}
    mock_request.return_value = mock_resp

    client = APIClient("https://api.example.com")
    res = client.post("/posts", data={"title": "New Post"})

    assert res["id"] == 101
    mock_request.assert_called_once_with(
        method="POST",
        url="https://api.example.com/posts",
        params=None,
        json={"title": "New Post"},
        timeout=10.0
    )

@patch("requests.Session.request")
def test_client_put_request(mock_request):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"id": 1, "title": "Updated"}
    mock_request.return_value = mock_resp

    client = APIClient("https://api.example.com")
    res = client.put("/posts/1", data={"title": "Updated"})
    assert res["title"] == "Updated"

@patch("requests.Session.request")
def test_client_patch_request(mock_request):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"id": 1, "title": "Patched"}
    mock_request.return_value = mock_resp

    client = APIClient("https://api.example.com")
    res = client.patch("/posts/1", data={"title": "Patched"})
    assert res["title"] == "Patched"

@patch("requests.Session.request")
def test_client_delete_request(mock_request):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {}
    mock_request.return_value = mock_resp

    client = APIClient("https://api.example.com")
    res = client.delete("/posts/1")
    assert res == {}

@patch("requests.Session.request")
def test_client_404_raises_not_found_error(mock_request):
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_request.return_value = mock_resp

    client = APIClient("https://api.example.com")
    with pytest.raises(APINotFoundError, match="was not found"):
        client.get("/invalid_endpoint")

@patch("requests.Session.request")
def test_client_timeout_raises_timeout_error(mock_request):
    mock_request.side_effect = requests.exceptions.Timeout("Connection timed out")

    client = APIClient("https://api.example.com")
    with pytest.raises(APITimeoutError, match="timed out"):
        client.get("/users")

@patch("requests.Session.request")
def test_client_connection_error_raises_exception(mock_request):
    mock_request.side_effect = requests.exceptions.ConnectionError("Refused")

    client = APIClient("https://api.example.com")
    with pytest.raises(APIConnectionError, match="Failed connecting"):
        client.get("/users")

@patch("requests.Session.request")
def test_bonus_sdk_resources_users_and_posts(mock_request):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"id": 1, "name": "Suraj"}]
    mock_request.return_value = mock_resp

    client = APIClient("https://api.example.com")
    users = client.users.list()
    assert len(users) == 1

class TestClientRunner(unittest.TestCase):
    def test_client_standalone(self):
        client = APIClient("https://api.example.com")
        self.assertIsNotNone(client)

if __name__ == "__main__":
    unittest.main()
