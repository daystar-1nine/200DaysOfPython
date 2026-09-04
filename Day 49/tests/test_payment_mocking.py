# ==============================================================================
# Program    : External Service Mocking Unit & Integration Tests (test_payment_mocking.py)
# Objective  : Test credit card payment endpoint using unittest.mock.patch and MagicMock.
# Concept    : External Service Mocking (unittest.mock)
# Why Used   : Ensures third-party payment APIs are never called during automated test suite execution.
# ==============================================================================

import os
import sys
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.payment_service import PaymentService, PaymentGatewayClient
from app.exceptions import PaymentGatewayError

def test_payment_service_unit_with_mock_client():
    """Unit test PaymentService using MagicMock for PaymentGatewayClient."""
    mock_client = MagicMock(spec=PaymentGatewayClient)
    mock_client.charge.return_value = {
        "transaction_id": "tx_mock_12345",
        "status": "success",
        "amount": 150.0,
        "currency": "USD"
    }

    service = PaymentService(client=mock_client)
    from app.schemas.payment import PaymentChargeRequest

    req = PaymentChargeRequest(amount=150.0, currency="USD", card_number="4111111111111111")
    response = service.process_payment(req)

    assert response.transaction_id == "tx_mock_12345"
    assert response.status == "success"
    assert response.amount == 150.0
    mock_client.charge.assert_called_once_with(
        amount=150.0,
        currency="USD",
        card_number="4111111111111111"
    )

def test_payment_endpoint_with_unittest_patch(client, normal_user_headers):
    """Integration test POST /payments/charge using unittest.mock.patch to intercept Gateway call."""
    mock_response = {
        "transaction_id": "tx_patched_9999",
        "status": "success",
        "amount": 250.0,
        "currency": "USD"
    }

    with patch("app.services.payment_service.PaymentGatewayClient.charge", return_value=mock_response) as mock_charge:
        payload = {
            "amount": 250.0,
            "currency": "USD",
            "card_number": "4111111111111111"
        }
        response = client.post("/payments/charge", json=payload, headers=normal_user_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["transaction_id"] == "tx_patched_9999"
        assert data["status"] == "success"
        assert data["amount"] == 250.0

        # Assert external charge method was called with exact request parameters
        mock_charge.assert_called_once_with(
            amount=250.0,
            currency="USD",
            card_number="4111111111111111"
        )

def test_payment_endpoint_handles_declined_card_mock(client, normal_user_headers):
    """Test payment endpoint returns HTTP 502 when payment gateway raises PaymentGatewayError."""
    with patch("app.services.payment_service.PaymentGatewayClient.charge", side_effect=PaymentGatewayError("Card declined")):
        payload = {
            "amount": 100.0,
            "currency": "USD",
            "card_number": "0000000000000000"
        }
        response = client.post("/payments/charge", json=payload, headers=normal_user_headers)
        assert response.status_code == 502
        data = response.json()
        assert data["error"]["code"] == "PAYMENT_FAILED"

def test_payment_endpoint_unauthenticated_returns_401(client):
    """Test charge endpoint without auth headers returns HTTP 401 Unauthorized."""
    payload = {"amount": 100.0, "currency": "USD", "card_number": "4111111111111111"}
    response = client.post("/payments/charge", json=payload)
    assert response.status_code == 401
