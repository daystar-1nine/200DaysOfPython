# ==============================================================================
# Program    : Pydantic Product Schemas (product.py)
# Objective  : Define ProductCreate and ProductResponse schemas.
# Concept    : Pydantic Data Modeling
# Why Used   : Validates input payloads for /products resource router.
# ==============================================================================

from pydantic import BaseModel, Field

class ProductCreate(BaseModel):
    title: str = Field(..., min_length=2, max_length=100)
    price: float = Field(..., gt=0.0)

class ProductResponse(BaseModel):
    id: int
    title: str
    price: float
