# ==============================================================================
# Program    : Payment Service & Mockable Gateway Integration (payment_service.py)
# Objective  : Provide credit card processing interface intended for external API mocking tests.
# Concept    : External Service Integration & Mocking Target
# Why Used   : Demonstrates mocking third-party payment services during test execution.
# ==============================================================================

import uuid
from app.config import settings
from app.schemas.payment import PaymentChargeRequest, PaymentResponse
from app.exceptions import PaymentGatewayError

class PaymentGatewayClient:
    """Simulates external payment gateway API client."""
    def __init__(self, api_url: str = settings.PAYMENT_GATEWAY_URL):
        self.api_url = api_url

    def charge(self, amount: float, currency: str, card_number: str) -> dict:
        """Call external credit card payment processor.

        In production, this makes an HTTP POST request to Stripe/PayPal.
        In test suites, this method is intercepted using `unittest.mock.patch`!
        """
        if amount <= 0:
            raise ValueError("Charge amount must be positive.")
        if card_number == "0000000000000000":
            raise PaymentGatewayError("Card declined by issuing bank.")

        tx_id = f"tx_{uuid.uuid4().hex[:12]}"
        return {
            "transaction_id": tx_id,
            "status": "success",
            "amount": amount,
            "currency": currency
        }

class PaymentService:
    def __init__(self, client: PaymentGatewayClient = None):
        self.client = client or PaymentGatewayClient()

    def process_payment(self, req: PaymentChargeRequest) -> PaymentResponse:
        res_data = self.client.charge(
            amount=req.amount,
            currency=req.currency,
            card_number=req.card_number
        )
        return PaymentResponse(
            transaction_id=res_data["transaction_id"],
            status=res_data["status"],
            amount=res_data["amount"],
            currency=res_data["currency"]
        )
