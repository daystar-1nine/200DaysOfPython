"""
===============================================================================
DAY 50 — USER SCHEMAS (PYDANTIC DTO DEFINITIONS)
===============================================================================
This module defines Pydantic validation schemas for user registration, user
profile response payloads, login credentials, and JWT token representations.
===============================================================================
"""

from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, ConfigDict


class UserCreate(BaseModel):
    """Schema for user registration request body."""

    # What is used: Pydantic Field validation attributes.
    # Why it is used: Validates user input before passing payload to registration service.
    # How it works: Ensures name length is >= 2, email is valid email format, and password length >= 6.
    name: str = Field(..., min_length=2, max_length=100, examples=["Suraj Sawant"])
    email: EmailStr = Field(..., examples=["suraj@example.com"])
    password: str = Field(..., min_length=6, max_length=100, examples=["secret123"])
    role: str = Field(default="user", examples=["user", "admin"])


class UserLogin(BaseModel):
    """Schema for user authentication credentials body."""

    email: EmailStr = Field(..., examples=["suraj@example.com"])
    password: str = Field(..., examples=["secret123"])


class UserResponse(BaseModel):
    """Schema for public user profile serialization."""

    # What is used: ORM serialization configuration model_config.
    # Why it is used: Allows Pydantic to read attributes directly from SQLAlchemy User ORM instances.
    # How it works: Sets from_attributes = True.
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: EmailStr
    role: str
    created_at: datetime


class Token(BaseModel):
    """Schema for JWT authentication response."""

    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Schema for decoded JWT token payload claims."""

    sub: str | None = None
    role: str | None = None
