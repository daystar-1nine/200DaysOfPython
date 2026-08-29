"""App Dependencies Package Initialization."""
import os
import sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.dependencies.auth import get_current_user
from app.dependencies.providers import get_user_repository, get_user_service, get_product_repository, get_product_service

__all__ = [
    "get_current_user",
    "get_user_repository",
    "get_user_service",
    "get_product_repository",
    "get_product_service"
]
