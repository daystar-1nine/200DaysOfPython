# ==============================================================================
# Program    : Mini E-Commerce Backend Composition Root (main.py)
# Objective  : Auto-create database tables (Base.metadata.create_all), register routers and custom exception handlers.
# Concept    : Composition Root & Table Schema Initialization
# Why Used   : Initializes database schemas and starts ASGI application server.
# ==============================================================================

import os
import sys
from fastapi import FastAPI

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import engine, Base
# Explicitly import all ORM model classes so they register on Base.metadata
from app.models.user import User  # noqa: F401
from app.models.product import Product  # noqa: F401
from app.models.order import Order  # noqa: F401
from app.models.order_item import OrderItem  # noqa: F401

from app.exceptions import ECommerceAPIError, ecommerce_exception_handler
from app.routers.users import router as users_router
from app.routers.products import router as products_router
from app.routers.orders import router as orders_router

# Auto-create all database tables on application startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Mini E-Commerce Backend",
    description="Relational E-Commerce API with FastAPI, SQLAlchemy 2.0, Transactions, and Eager Loading.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Register Custom Exception Handler
app.add_exception_handler(ECommerceAPIError, ecommerce_exception_handler)

# Register Modular Resource Routers
app.include_router(users_router)
app.include_router(products_router)
app.include_router(orders_router)

@app.get("/", tags=["General"])
def read_root():
    return {
        "message": "Welcome to Mini E-Commerce Backend API",
        "architecture": "User -> Orders -> OrderItems -> Product",
        "documentation": "/docs"
    }

@app.get("/health", tags=["General"])
def health_check():
    return {"status": "healthy", "database": "connected"}

@app.get("/about", tags=["General"])
def about_api():
    return {
        "application": "Mini E-Commerce Backend",
        "version": "1.0.0",
        "framework": "FastAPI",
        "orm": "SQLAlchemy 2.0",
        "author": "Suraj Sawant",
        "repository": "https://github.com/daystar-1nine/200DaysOfPython"
    }
