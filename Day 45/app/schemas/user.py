# ==============================================================================
# Program    : User Schemas (user.py)
# Objective  : Define Pydantic request and response schemas including nested Order responses.
# Concept    : Nested Pydantic Schemas (Day 45 requirement)
# Why Used   : Formats user details and nested list of orders as JSON.
# ==============================================================================

import os
import sys
from pydantic import BaseModel, Field, ConfigDict

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.schemas.order import OrderResponse

class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)
    email: str

class UserResponse(BaseModel):
    id: int
    name: str
    email: str

    model_config = ConfigDict(from_attributes=True)

# What is used : UserWithOrdersResponse (Nested Pydantic Schema)
# Why it is used: Serializes user entity along with nested list of OrderResponse objects
class UserWithOrdersResponse(BaseModel):
    id: int
    name: str
    email: str
    orders: list[OrderResponse] = []

    model_config = ConfigDict(from_attributes=True)
