"""
===============================================================================
DAY 50 — SQLALCHEMY ORM MODELS PACKAGE
===============================================================================
This package exports SQLAlchemy models (User, Task) for registry import.
===============================================================================
"""

from app.models.user import User
from app.models.task import Task

__all__ = ["User", "Task"]
