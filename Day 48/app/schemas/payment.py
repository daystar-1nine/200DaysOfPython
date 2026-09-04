# ==============================================================================
# Program    : Payment Pydantic Schemas (payment.py)
# Objective  : Define Payment charge request and response models for external gateway testing.
# Concept    : Payment API Data Transfer Object
# Why Used   : Structures payment requests processed by external payment services.
# ==============================================================================

from typing import Optional
from pydantic import BaseModel, Field

class PaymentChargeRequest(BaseModel):
    amount: float = Field(..., gt=0.0, description="Amount to charge credit card")
    currency: str = Field("USD", min_length=3, max_length=3, description="ISO currency code")
    card_number: str = Field(..., min_length=15, max_length=16, description="16-digit credit card number")

class PaymentResponse(BaseModel):
    transaction_id: str
    status: str
    amount: float
    currency: str
