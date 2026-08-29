"""App Routers Package Initialization."""
import os
import sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, sys.path)

from app.routers.users import router as users_router
from app.routers.products import router as products_router

__all__ = ["users_router", "products_router"]
