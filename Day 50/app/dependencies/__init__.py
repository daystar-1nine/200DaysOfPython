"""
===============================================================================
DAY 50 — DEPENDENCIES PACKAGE
===============================================================================
This package exports authentication and authorization dependency functions.
===============================================================================
"""

from app.dependencies.auth import get_current_user, require_admin

__all__ = ["get_current_user", "require_admin"]
