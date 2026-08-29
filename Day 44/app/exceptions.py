# ==============================================================================
# Program    : Domain Exceptions Taxonomy (exceptions.py)
# Objective  : Define custom domain exception classes and FastAPI exception handlers.
# Concept    : Exception Handling & HTTP Mapping
# Why Used   : Maps database and domain errors to structured JSON responses.
# ==============================================================================

from fastapi import Request, status
from fastapi.responses import JSONResponse

class UserAPIError(Exception):
    """Base Exception for API errors."""
    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

class UserNotFoundError(UserAPIError):
    def __init__(self, user_id: int):
        super().__init__(f"User with ID #{user_id} was not found.", status_code=status.HTTP_404_NOT_FOUND)

class UserAlreadyExistsError(UserAPIError):
    def __init__(self, email: str):
        super().__init__(f"User with email '{email}' already exists.", status_code=status.HTTP_409_CONFLICT)

class DatabaseError(UserAPIError):
    def __init__(self, detail: str):
        super().__init__(f"Database Error: {detail}", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

async def user_api_exception_handler(request: Request, exc: UserAPIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "error_type": exc.__class__.__name__}
    )
