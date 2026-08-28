# ==============================================================================
# Test Suite : API Client Pytest Suite with Mocks (Advanced Challenge)
# Objective  : Mock HTTP requests using @patch to test 200, 401, 404, 429, 500, Timeout, and Retry logic.
# Concept    : API Mocking with unittest.mock.patch & Pytest Assertions
# Why Used   : Simulates real-world server failures without making real network calls.
# ==============================================================================

import os
import sys
import unittest
from unittest.mock import patch, Mock
import pytest
import requests

# Append Day 27 github_cli package path
pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Day 27", "github_cli"))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from api_client import APIClient

@patch("api_client.requests.Session.get")
def test_api_200_success(mock_get):
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"login": "octocat", "name": "The Octocat"}
    mock_get.return_value = mock_resp

    client = APIClient(use_cache=False)
    result = client.get("/users/octocat", use_cache=False)
    assert result["login"] == "octocat"
    assert result["name"] == "The Octocat"

@patch("api_client.requests.Session.get")
def test_api_404_not_found(mock_get):
    mock_resp = Mock()
    mock_resp.status_code = 404
    mock_resp.raise_for_status.side_effect = requests.HTTPError("404 Client Error: Not Found")
    mock_get.return_value = mock_resp

    client = APIClient()
    with pytest.raises(requests.HTTPError):
        client.get("/users/invalid_user_99999", use_cache=False)

@patch("api_client.requests.Session.get")
def test_api_timeout_retry_recovery(mock_get):
    # Simulate attempt 1 timeout, attempt 2 success
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "recovered"}

    mock_get.side_effect = [requests.Timeout("Request timed out"), mock_resp]

    client = APIClient(max_retries=2)
    with patch("time.sleep", return_value=None):
        result = client.get("/test", use_cache=False)
    assert result["status"] == "recovered"
    assert mock_get.call_count == 2

class TestAPIClientMockRunner(unittest.TestCase):
    @patch("api_client.requests.Session.get")
    def test_api_mocks_standalone(self, mock_get):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"login": "octocat"}
        mock_get.return_value = mock_resp

        client = APIClient()
        res = client.get("/users/octocat", use_cache=False)
        self.assertEqual(res["login"], "octocat")

if __name__ == "__main__":
    unittest.main()
