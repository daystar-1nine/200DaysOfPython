# ==============================================================================
# Program    : Authentication Pydantic Schemas (auth.py)
# Objective  : Define RegisterRequest, LoginRequest, TokenResponse, and TokenData schemas.
# Concept    : Pydantic Data Validation & Serialization
# Why Used   : Validates user registration payloads, credentials, and structures access tokens.
# ==============================================================================

from typing import Optional
from pydantic import BaseModel, EmailStr, Field

class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Full name of user")
    email: EmailStr = Field(..., description="Unique email address")
    password: str = Field(..., min_length=6, description="Plaintext password (min 6 characters)")
    age: Optional[int] = Field(None, ge=0, le=150, description="User age in years")
    phone: Optional[str] = Field(None, max_length=20, description="Contact phone number")
    role: Optional[str] = Field("user", description="User authorization role (user or admin)")

class LoginRequest(BaseModel):
    email: EmailStr = Field(..., description="Registered user email")
    password: str = Field(..., description="Plaintext password candidate")

class TokenResponse(BaseModel):
    access_token: str = Field(..., description="Signed JWT access token")
    token_type: str = Field("bearer", description="Token standard type")

class TokenData(BaseModel):
    user_id: Optional[int] = None
    email: Optional[str] = None
    role: Optional[str] = None
