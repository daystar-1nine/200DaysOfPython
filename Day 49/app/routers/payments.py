# ==============================================================================
# Program    : Payment Processing Router Module (payments.py)
# Objective  : Route handler for POST /payments/charge.
# Concept    : External Service Integration Endpoint
# Why Used   : Exposes credit card processing endpoint for external API mocking tests.
# ==============================================================================

from fastapi import APIRouter, Depends, status
from app.models.user import User
from app.schemas.payment import PaymentChargeRequest, PaymentResponse
from app.dependencies.auth import get_current_user
from app.services.payment_service import PaymentService

router = APIRouter(prefix="/payments", tags=["Payments"])

@router.post(
    "/charge",
    response_model=PaymentResponse,
    status_code=status.HTTP_200_OK,
    summary="Process credit card payment",
    description="Authenticated endpoint processing credit card payments via external payment gateway."
)
def charge_card(
    req: PaymentChargeRequest,
    current_user: User = Depends(get_current_user)
):
    """Authenticated endpoint processing credit card payments via external Gateway."""
    service = PaymentService()
    return service.process_payment(req)
