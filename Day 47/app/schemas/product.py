# ==============================================================================
# Program    : Product Pydantic Schemas (product.py)
# Objective  : Define Product creation, update, and response Pydantic models.
# Concept    : Pydantic Data Validation & Serialization
# Why Used   : Validates product catalog creation and partial updates.
# ==============================================================================

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

class ProductBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=150)
    price: float = Field(..., gt=0.0)
    stock: int = Field(..., ge=0)
    description: Optional[str] = None
    category: Optional[str] = None

class ProductCreate(ProductBase):
    pass

class ProductUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=150)
    price: Optional[float] = Field(None, gt=0.0)
    stock: Optional[int] = Field(None, ge=0)
    description: Optional[str] = None
    category: Optional[str] = None

class ProductResponse(ProductBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
