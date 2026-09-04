# ==============================================================================
# Program    : Domain Exceptions Taxonomy (exceptions.py)
# Objective  : Custom domain exception hierarchy mapped to HTTP 400, 401, 403, 404, 409, and 502 status codes.
# Concept    : Security & Exception Domain Mapping
# Why Used   : Encapsulates domain error handling for authentication, payments, and orders.
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

class InvalidCredentialsError(ECommerceAPIError):
    def __init__(self):
        super().__init__("Invalid email or password.", status_code=status.HTTP_401_UNAUTHORIZED)

class AuthenticationRequiredError(ECommerceAPIError):
    def __init__(self, detail: str = "Could not validate authentication credentials."):
        super().__init__(detail, status_code=status.HTTP_401_UNAUTHORIZED)

class PermissionDeniedError(ECommerceAPIError):
    def __init__(self, detail: str = "You do not have permission to perform this action."):
        super().__init__(detail, status_code=status.HTTP_403_FORBIDDEN)

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

class PaymentGatewayError(ECommerceAPIError):
    def __init__(self, reason: str):
        super().__init__(f"Payment processing failed: {reason}", status_code=status.HTTP_502_BAD_GATEWAY)

async def ecommerce_exception_handler(request: Request, exc: ECommerceAPIError) -> JSONResponse:
    headers = {}
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "error_type": exc.__class__.__name__},
        headers=headers
    )
