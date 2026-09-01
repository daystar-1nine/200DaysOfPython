# ==============================================================================
# Program    : User Schemas (user.py)
# Objective  : Define Pydantic user request and response models supporting phone & created_at fields.
# Concept    : Pydantic Schemas with Optional Migrated Attributes
# Why Used   : Validates user input and formats API JSON responses.
# ==============================================================================

import os
import sys
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.schemas.order import OrderResponse

class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: str
    phone: Optional[str] = Field(default=None, max_length=20)

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    phone: Optional[str] = None
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

class UserWithOrdersResponse(BaseModel):
    id: int
    name: str
    email: str
    phone: Optional[str] = None
    created_at: Optional[datetime] = None
    orders: list[OrderResponse] = []

    model_config = ConfigDict(from_attributes=True)
