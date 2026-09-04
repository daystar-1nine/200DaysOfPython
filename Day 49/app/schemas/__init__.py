"""Pydantic Schemas Initialization."""
from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse, TokenData
from app.schemas.user import UserCreate, UserResponse, UserWithOrdersResponse
from app.schemas.product import ProductCreate, ProductUpdate, ProductResponse
from app.schemas.order import OrderItemCreate, OrderItemResponse, OrderCreate, OrderResponse
from app.schemas.payment import PaymentChargeRequest, PaymentResponse
from app.schemas.errors import APIErrorDetail, APIErrorResponse

__all__ = [
    "RegisterRequest",
    "LoginRequest",
    "TokenResponse",
    "TokenData",
    "UserCreate",
    "UserResponse",
    "UserWithOrdersResponse",
    "ProductCreate",
    "ProductUpdate",
    "ProductResponse",
    "OrderItemCreate",
    "OrderItemResponse",
    "OrderCreate",
    "OrderResponse",
    "PaymentChargeRequest",
    "PaymentResponse",
    "APIErrorDetail",
    "APIErrorResponse"
]
