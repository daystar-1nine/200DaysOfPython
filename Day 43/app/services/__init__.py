"""App Services Package Initialization."""
import os
import sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, sys.path)

from app.services.user_service import UserService
from app.services.product_service import ProductService

__all__ = ["UserService", "ProductService"]
