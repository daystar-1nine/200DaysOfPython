# ==============================================================================
# Test Suite : Unit Tests for Currency API Client with Mocks
# Objective  : Mock HTTP requests using @patch to test exchange rate API fetching and caching.
# Concept    : API Mocking & Isolation
# Why Used   : Asserts API client behavior without making live network calls.
# ==============================================================================

import os
import sys
import unittest
from unittest.mock import patch, Mock

base_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
pkg_root = os.path.join(base_repo, "Day 30", "pyfinance", "src")
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from pyfinance.api.client import CurrencyAPIClient

@patch("pyfinance.api.client.CurrencyAPIClient._read_cache", return_value=None)
@patch("pyfinance.api.client.requests.Session.get")
def test_fetch_exchange_rate_mock(mock_get, mock_cache):
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "result": "success",
        "base_code": "USD",
        "rates": {"INR": 83.50, "EUR": 0.92}
    }
    mock_get.return_value = mock_resp

    client = CurrencyAPIClient()
    rate = client.fetch_exchange_rate("USD", "INR")
    assert rate == 83.50

class TestAPIClientMockRunner(unittest.TestCase):
    @patch("pyfinance.api.client.CurrencyAPIClient._read_cache", return_value=None)
    @patch("pyfinance.api.client.requests.Session.get")
    def test_fetch_rate_standalone(self, mock_get, mock_cache):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "result": "success",
            "base_code": "USD",
            "rates": {"INR": 83.50}
        }
        mock_get.return_value = mock_resp

        client = CurrencyAPIClient()
        rate = client.fetch_exchange_rate("USD", "INR")
        self.assertEqual(rate, 83.50)

if __name__ == "__main__":
    unittest.main()
