# ==============================================================================
# Program    : Pydantic User Schemas (user.py)
# Objective  : Define Pydantic request and response schemas separated from SQLAlchemy models.
# Concept    : Pydantic Schemas vs SQLAlchemy Models (Day 44 requirement)
# Why Used   : Validates HTTP API payloads and formats JSON responses.
# ==============================================================================

from pydantic import BaseModel, Field, ConfigDict

class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=50, description="Full Name")
    email: str = Field(..., description="Email Address")
    age: int = Field(..., gt=0, lt=120, description="Age in Years")

class UserUpdate(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)
    email: str
    age: int = Field(..., gt=0, lt=120)

class UserPatch(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=50)
    email: str | None = Field(default=None)
    age: int | None = Field(default=None, gt=0, lt=120)

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    age: int

    model_config = ConfigDict(from_attributes=True)
