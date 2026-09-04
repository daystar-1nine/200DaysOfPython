"""
===============================================================================
DAY 50 — TASK SCHEMAS (PYDANTIC DTO DEFINITIONS & ENUMS)
===============================================================================
This module defines Python Enums for TaskStatus and TaskPriority, alongside
Pydantic validation schemas for Task creation, full update, partial patch, and response serialization.
===============================================================================
"""

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class TaskStatus(str, Enum):
    """Enumeration of valid task progress states."""

    TODO = "TODO"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


class TaskPriority(str, Enum):
    """Enumeration of task priority levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TaskCreate(BaseModel):
    """Schema for task creation request body."""

    # What is used: Field constraints and string enum default assignments.
    # Why it is used: Validates task payload parameters prior to DB insert.
    # How it works: Ensures title is non-empty string and validates status/priority values.
    title: str = Field(..., min_length=1, max_length=200, examples=["Complete Day 50 API"])
    description: str | None = Field(None, max_length=2000, examples=["Implement TaskFlow API backend and pytest suite."])
    status: TaskStatus = Field(default=TaskStatus.TODO, examples=[TaskStatus.TODO])
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, examples=[TaskPriority.MEDIUM])
    due_date: datetime | None = Field(None, examples=["2026-12-31T23:59:59Z"])


class TaskUpdate(BaseModel):
    """Schema for full task replacement (PUT)."""

    title: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)
    status: TaskStatus
    priority: TaskPriority
    due_date: datetime | None = None


class TaskPatch(BaseModel):
    """Schema for partial task modification (PATCH)."""

    title: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)
    status: TaskStatus | None = None
    priority: TaskPriority | None = None
    due_date: datetime | None = None


class TaskResponse(BaseModel):
    """Schema for task entity response serialization."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    description: str | None
    status: TaskStatus
    priority: TaskPriority
    due_date: datetime | None
    user_id: int
    created_at: datetime
    updated_at: datetime
