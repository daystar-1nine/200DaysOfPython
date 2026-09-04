# ==============================================================================
# Program    : Standardized Error Pydantic Schemas (errors.py)
# Objective  : Define APIErrorDetail and APIErrorResponse models for predictable error payloads.
# Concept    : API Response Consistency & OpenAPI Documentation
# Why Used   : Guarantees every HTTP error returns a uniform JSON structure.
# ==============================================================================

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class APIErrorDetail(BaseModel):
    code: str = Field(..., description="Machine-readable uppercase error code (e.g. INSUFFICIENT_STOCK)")
    message: str = Field(..., description="Human-readable explanation of error")
    request_id: Optional[str] = Field(None, description="Unique correlation ID assigned to request")
    fields: Optional[Dict[str, Any]] = Field(None, description="Optional field-level validation errors")

class APIErrorResponse(BaseModel):
    error: APIErrorDetail = Field(..., description="Standardized error wrapper object")
