"""App Routes Package Initialization."""
import os
import sys

pkg_root = os.path.abspath(os.path.dirname(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from users import router as users_router

__all__ = ["users_router"]
