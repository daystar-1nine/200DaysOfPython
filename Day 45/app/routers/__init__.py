"""App Routers Package Initialization."""
import os
import sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.routers.users import router as users_router
from app.routers.products import router as products_router
from app.routers.orders import router as orders_router

__all__ = ["users_router", "products_router", "orders_router"]
