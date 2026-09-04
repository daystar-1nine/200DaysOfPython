# ==============================================================================
# Program    : Order Pydantic Schemas (order.py)
# Objective  : Define OrderItem and Order creation and response models.
# Concept    : Nested Pydantic Schema Serialization
# Why Used   : Formats order checkout requests and detailed nested order responses.
# ==============================================================================

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from app.schemas.product import ProductResponse

class OrderItemCreate(BaseModel):
    product_id: int = Field(..., gt=0, description="Product ID to purchase")
    quantity: int = Field(1, gt=0, description="Quantity of units")

class OrderItemResponse(BaseModel):
    id: int
    product_id: int
    quantity: int
    price: float
    product: Optional[ProductResponse] = None

    model_config = ConfigDict(from_attributes=True)

class OrderCreate(BaseModel):
    items: List[OrderItemCreate] = Field(..., min_length=1, description="List of line items")

class OrderResponse(BaseModel):
    id: int
    user_id: int
    total_amount: float
    status: str
    created_at: datetime
    items: List[OrderItemResponse] = []

    model_config = ConfigDict(from_attributes=True)
