"""App Repositories Package Initialization."""
import os
import sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.repositories.user_repository import UserRepository
from app.repositories.product_repository import ProductRepository

__all__ = ["UserRepository", "ProductRepository"]
