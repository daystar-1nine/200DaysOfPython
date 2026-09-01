# ==============================================================================
# Program    : Product Schemas (product.py)
# Objective  : Define Product Pydantic schemas with description, category, and created_at.
# Concept    : Product Validation & Data Transfer Objects
# Why Used   : Validates product inputs and inventory stock.
# ==============================================================================

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

class ProductCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    price: float = Field(..., gt=0.0)
    stock: int = Field(..., ge=0)
    description: Optional[str] = None
    category: Optional[str] = Field(default=None, max_length=50)

class ProductUpdate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    price: float = Field(..., gt=0.0)
    stock: int = Field(..., ge=0)
    description: Optional[str] = None
    category: Optional[str] = Field(default=None, max_length=50)

class ProductPatch(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=100)
    price: Optional[float] = Field(default=None, gt=0.0)
    stock: Optional[int] = Field(default=None, ge=0)
    description: Optional[str] = None
    category: Optional[str] = Field(default=None, max_length=50)

class ProductResponse(BaseModel):
    id: int
    name: str
    price: float
    stock: int
    description: Optional[str] = None
    category: Optional[str] = None
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
