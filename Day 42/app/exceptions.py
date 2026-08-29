# ==============================================================================
# Program    : Application Domain Exceptions (exceptions.py)
# Objective  : Define domain exception classes (UserNotFoundError, UserAlreadyExistsError) and FastAPI handlers.
# Concept    : Exception Domain Hierarchy & FastAPI Handler Mapping
# Why Used   : Decouples business exception raising from HTTP response status formatting.
# ==============================================================================

from fastapi import Request, status
from fastapi.responses import JSONResponse

class UserAPIError(Exception):
    """Base Exception for User Management API errors."""
    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

class UserNotFoundError(UserAPIError):
    """Raised when user record ID is not found."""
    def __init__(self, user_id: int):
        super().__init__(f"User with ID #{user_id} was not found.", status_code=status.HTTP_404_NOT_FOUND)
        self.user_id = user_id

class UserAlreadyExistsError(UserAPIError):
    """Raised when user email already exists in system."""
    def __init__(self, email: str):
        super().__init__(f"User with email '{email}' already exists.", status_code=status.HTTP_409_CONFLICT)
        self.email = email

async def user_api_exception_handler(request: Request, exc: UserAPIError) -> JSONResponse:
    """FastAPI Exception Handler converting UserAPIError to JSONResponse."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "error_type": exc.__class__.__name__}
    )
