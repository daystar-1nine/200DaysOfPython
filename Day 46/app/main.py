# ==============================================================================
# Program    : Mini E-Commerce Backend V2 Composition Root (main.py)
# Objective  : Register routers and custom exception handlers.
# Concept    : Composition Root & Application Entrypoint
# Why Used   : Starts ASGI application server (schema migrations handled via Alembic).
# ==============================================================================

import os
import sys
from fastapi import FastAPI

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.models.user import User  # noqa: F401
from app.models.product import Product  # noqa: F401
from app.models.order import Order  # noqa: F401
from app.models.order_item import OrderItem  # noqa: F401

from app.exceptions import ECommerceAPIError, ecommerce_exception_handler
from app.routers.users import router as users_router
from app.routers.products import router as products_router
from app.routers.orders import router as orders_router

app = FastAPI(
    title="Mini E-Commerce Backend V2",
    description="Relational E-Commerce API managed via Alembic Migrations, FastAPI, SQLAlchemy 2.0.",
    version="2.0.0",
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
        "message": "Welcome to Mini E-Commerce Backend V2 API",
        "migrations": "Managed via Alembic",
        "documentation": "/docs"
    }

@app.get("/health", tags=["General"])
def health_check():
    return {"status": "healthy", "database": "connected"}

@app.get("/about", tags=["General"])
def about_api():
    return {
        "application": "Mini E-Commerce Backend V2",
        "version": "2.0.0",
        "framework": "FastAPI",
        "orm": "SQLAlchemy 2.0",
        "migration_tool": "Alembic",
        "author": "Suraj Sawant",
        "repository": "https://github.com/daystar-1nine/200DaysOfPython"
    }
