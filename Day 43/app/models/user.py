# ==============================================================================
# Program    : Pydantic User Schemas (user.py)
# Objective  : Define Pydantic request, response, and profile models.
# Concept    : Pydantic Data Validation Models (Day 39)
# Why Used   : Enforces input validation rules and output schema definitions.
# ==============================================================================

from pydantic import BaseModel, Field

class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=50, description="Full Name")
    email: str = Field(..., description="Email Address")
    age: int = Field(..., gt=0, lt=120, description="Age in Years")

class UserUpdate(BaseModel):
    name: str = Field(..., min_length=2, max_length=50, description="Full Name")
    email: str = Field(..., description="Email Address")
    age: int = Field(..., gt=0, lt=120, description="Age in Years")

class UserPatch(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=50)
    email: str | None = Field(default=None)
    age: int | None = Field(default=None, gt=0, lt=120)

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    age: int

class UserProfileResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str = "user"
