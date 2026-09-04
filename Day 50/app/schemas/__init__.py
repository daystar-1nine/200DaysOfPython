"""
===============================================================================
DAY 50 — PYDANTIC SCHEMAS PACKAGE
===============================================================================
This package exports Pydantic models for validation and serialization contracts.
===============================================================================
"""

from app.schemas.user import UserCreate, UserResponse, UserLogin, Token, TokenData
from app.schemas.task import TaskCreate, TaskUpdate, TaskPatch, TaskResponse, TaskStatus, TaskPriority
from app.schemas.errors import ErrorPayload, ErrorDetail

__all__ = [
    "UserCreate",
    "UserResponse",
    "UserLogin",
    "Token",
    "TokenData",
    "TaskCreate",
    "TaskUpdate",
    "TaskPatch",
    "TaskResponse",
    "TaskStatus",
    "TaskPriority",
    "ErrorPayload",
    "ErrorDetail",
]
