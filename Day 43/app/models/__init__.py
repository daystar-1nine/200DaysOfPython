"""App Models Package Initialization."""
import os
import sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.models.user import UserCreate, UserUpdate, UserPatch, UserResponse, UserProfileResponse
from app.models.product import ProductCreate, ProductResponse

__all__ = [
    "UserCreate",
    "UserUpdate",
    "UserPatch",
    "UserResponse",
    "UserProfileResponse",
    "ProductCreate",
    "ProductResponse"
]
