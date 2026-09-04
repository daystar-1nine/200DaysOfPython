# ==============================================================================
# Program    : User Pydantic Schemas (user.py)
# Objective  : Define User response models excluding sensitive password_hash fields.
# Concept    : Pydantic Response Serialization & Field Sanitization
# Why Used   : Ensures user credentials (password_hash) are NEVER exposed over public API endpoints.
# ==============================================================================

from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from pydantic import BaseModel, EmailStr, ConfigDict

if TYPE_CHECKING:
    from app.schemas.order import OrderResponse

class UserBase(BaseModel):
    name: str
    email: EmailStr
    age: Optional[int] = None
    phone: Optional[str] = None
    role: str = "user"

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class UserWithOrdersResponse(UserResponse):
    orders: List["OrderResponse"] = []

    model_config = ConfigDict(from_attributes=True)
