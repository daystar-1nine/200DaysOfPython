"""
===============================================================================
DAY 50 — REPOSITORIES PACKAGE
===============================================================================
This package exports data access repository classes (UserRepository, TaskRepository).
===============================================================================
"""

from app.repositories.user_repository import UserRepository
from app.repositories.task_repository import TaskRepository

__all__ = ["UserRepository", "TaskRepository"]
