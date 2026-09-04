# ==============================================================================
# Program    : Domain Custom Exceptions Taxonomy (exceptions.py)
# Objective  : Custom exception hierarchy mapped to HTTP status codes and standardized error codes.
# Concept    : Separation of Business Exceptions & Standardized Error Codes
# Why Used   : Decouples business logic from HTTP status code handling and error formatting.
# ==============================================================================

from fastapi import status

class ECommerceAPIError(Exception):
    """Base class for all domain custom exceptions."""
    def __init__(self, message: str, code: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code

class UserNotFoundError(ECommerceAPIError):
    def __init__(self, user_id: int):
        super().__init__(
            message=f"User #{user_id} was not found.",
            code="USER_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND
        )

class DuplicateEmailError(ECommerceAPIError):
    def __init__(self, email: str):
        super().__init__(
            message=f"User with email '{email}' already exists.",
            code="DUPLICATE_EMAIL",
            status_code=status.HTTP_409_CONFLICT
        )

class InvalidCredentialsError(ECommerceAPIError):
    def __init__(self):
        super().__init__(
            message="Invalid email or password credentials provided.",
            code="INVALID_CREDENTIALS",
            status_code=status.HTTP_401_UNAUTHORIZED
        )

class AuthenticationError(ECommerceAPIError):
    def __init__(self, detail: str = "Could not validate authentication credentials."):
        super().__init__(
            message=detail,
            code="AUTHENTICATION_FAILED",
            status_code=status.HTTP_401_UNAUTHORIZED
        )

class AuthorizationError(ECommerceAPIError):
    def __init__(self, detail: str = "Administrator privileges are required for this action."):
        super().__init__(
            message=detail,
            code="FORBIDDEN",
            status_code=status.HTTP_403_FORBIDDEN
        )

class ProductNotFoundError(ECommerceAPIError):
    def __init__(self, product_id: int):
        super().__init__(
            message=f"Product #{product_id} was not found.",
            code="PRODUCT_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND
        )

class OrderNotFoundError(ECommerceAPIError):
    def __init__(self, order_id: int):
        super().__init__(
            message=f"Order #{order_id} was not found.",
            code="ORDER_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND
        )

class InsufficientStockError(ECommerceAPIError):
    def __init__(self, product_title: str, requested: int, available: int):
        super().__init__(
            message=f"Insufficient stock for '{product_title}'. Requested: {requested}, Available: {available}.",
            code="INSUFFICIENT_STOCK",
            status_code=status.HTTP_409_CONFLICT
        )

class PaymentGatewayError(ECommerceAPIError):
    def __init__(self, reason: str):
        super().__init__(
            message=f"Payment processing failed: {reason}",
            code="PAYMENT_FAILED",
            status_code=status.HTTP_502_BAD_GATEWAY
        )
