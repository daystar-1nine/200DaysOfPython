"""
===============================================================================
DAY 50 — SERVICES PACKAGE
===============================================================================
This package exports application business logic services (AuthService, UserService, TaskService).
===============================================================================
"""

from app.services.auth_service import AuthService
from app.services.user_service import UserService
from app.services.task_service import TaskService

__all__ = ["AuthService", "UserService", "TaskService"]
