# ==============================================================================
# Program    : Order Schemas (order.py)
# Objective  : Define OrderItem and Order request/response Pydantic models.
# Concept    : Nested Pydantic Structuring for Orders & Order Items
# Why Used   : Parses POST /orders JSON payload containing nested product items.
# ==============================================================================

from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

class OrderItemCreate(BaseModel):
    product_id: int = Field(..., gt=0)
    quantity: int = Field(..., gt=0)

class OrderItemResponse(BaseModel):
    id: int
    product_id: int
    quantity: int
    price: float

    model_config = ConfigDict(from_attributes=True)

class OrderCreate(BaseModel):
    user_id: int = Field(..., gt=0)
    items: list[OrderItemCreate] = Field(..., min_length=1)

class OrderResponse(BaseModel):
    id: int
    user_id: int
    status: str
    total_amount: float
    created_at: datetime
    items: list[OrderItemResponse] = []

    model_config = ConfigDict(from_attributes=True)
