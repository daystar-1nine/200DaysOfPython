# ==============================================================================
# Program    : Pydantic User Schemas (user.py)
# Objective  : Define Pydantic request and response models with Field validations (Day 39 requirement).
# Concept    : Pydantic Data Models & Field Validation Rules
# Why Used   : Validates request payloads and filters response structures automatically.
# ==============================================================================

from pydantic import BaseModel, Field

class UserCreate(BaseModel):
    """Pydantic model for User creation request payload."""
    # What is used : Field(min_length=2, gt=0, lt=120)
    # Why it is used: Automatically enforces string length and numerical range validation
    name: str = Field(..., min_length=2, max_length=50, description="Full Name of User")
    email: str = Field(..., description="User Email Address")
    age: int = Field(..., gt=0, lt=120, description="User Age in Years")

class UserUpdate(BaseModel):
    """Pydantic model for PUT full resource replacement."""
    name: str = Field(..., min_length=2, max_length=50, description="Full Name of User")
    email: str = Field(..., description="User Email Address")
    age: int = Field(..., gt=0, lt=120, description="User Age in Years")

class UserPatch(BaseModel):
    """Pydantic model for PATCH partial resource update."""
    name: str | None = Field(default=None, min_length=2, max_length=50)
    email: str | None = Field(default=None)
    age: int | None = Field(default=None, gt=0, lt=120)

class UserResponse(BaseModel):
    """Pydantic model for API JSON responses."""
    id: int
    name: str
    email: str
    age: int
