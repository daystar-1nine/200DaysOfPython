# ==============================================================================
# Program    : Domain Exceptions Taxonomy (exceptions.py)
# Objective  : Custom domain exception hierarchy mapped to HTTP status codes.
# Concept    : Exception Handling Domain Mapping
# Why Used   : Provides clean exception classes for 404, 409, 400 bad requests.
# ==============================================================================

from fastapi import Request, status
from fastapi.responses import JSONResponse

class ECommerceAPIError(Exception):
    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

class UserNotFoundError(ECommerceAPIError):
    def __init__(self, user_id: int):
        super().__init__(f"User #{user_id} was not found.", status_code=status.HTTP_404_NOT_FOUND)

class UserAlreadyExistsError(ECommerceAPIError):
    def __init__(self, email: str):
        super().__init__(f"User with email '{email}' already exists.", status_code=status.HTTP_409_CONFLICT)

class ProductNotFoundError(ECommerceAPIError):
    def __init__(self, product_id: int):
        super().__init__(f"Product #{product_id} was not found.", status_code=status.HTTP_404_NOT_FOUND)

class OrderNotFoundError(ECommerceAPIError):
    def __init__(self, order_id: int):
        super().__init__(f"Order #{order_id} was not found.", status_code=status.HTTP_404_NOT_FOUND)

class InsufficientStockError(ECommerceAPIError):
    def __init__(self, product_title: str, requested: int, available: int):
        super().__init__(
            f"Insufficient stock for '{product_title}'. Requested: {requested}, Available: {available}.",
            status_code=status.HTTP_400_BAD_REQUEST
        )

async def ecommerce_exception_handler(request: Request, exc: ECommerceAPIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "error_type": exc.__class__.__name__}
    )
