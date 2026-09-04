"""
===============================================================================
DAY 50 — ROUTERS PACKAGE
===============================================================================
This package exports APIRouter instances for auth, users, tasks, admin, and health.
===============================================================================
"""

from app.routers.auth import router as auth_router
from app.routers.users import router as users_router
from app.routers.tasks import router as tasks_router
from app.routers.admin import router as admin_router
from app.routers.health import router as health_router

__all__ = ["auth_router", "users_router", "tasks_router", "admin_router", "health_router"]
