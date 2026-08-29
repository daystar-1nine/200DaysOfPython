# ==============================================================================
# Program    : Product Schemas (product.py)
# Objective  : Define ProductCreate, ProductUpdate, ProductPatch, and ProductResponse schemas.
# Concept    : Pydantic Data Validation
# Why Used   : Validates product inputs and inventory stock.
# ==============================================================================

from pydantic import BaseModel, Field, ConfigDict

class ProductCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    price: float = Field(..., gt=0.0)
    stock: int = Field(..., ge=0)

class ProductUpdate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    price: float = Field(..., gt=0.0)
    stock: int = Field(..., ge=0)

class ProductPatch(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=100)
    price: float | None = Field(default=None, gt=0.0)
    stock: int | None = Field(default=None, ge=0)

class ProductResponse(BaseModel):
    id: int
    name: str
    price: float
    stock: int

    model_config = ConfigDict(from_attributes=True)
