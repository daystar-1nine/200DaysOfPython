# ==============================================================================
# Program    : FastAPI Composition Root (main.py)
# Objective  : Instantiate FastAPI application, register routers, and configure exception handlers.
# Concept    : Composition Root & Application Entry Point
# Why Used   : Wires routers, CORS middleware, and domain exception handlers together.
# ==============================================================================

import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.config import settings
from app.exceptions import ECommerceAPIError, ecommerce_exception_handler
from app.routers import auth_router, users_router, products_router, orders_router, payments_router

app = FastAPI(
    title="Secure E-Commerce Backend V4",
    description="E-Commerce API with Auth, RBAC, Order Placement, Payment Gateway Mocking & Pytest Test Suite",
    version="4.0.0"
)

# Exception Handler Registration
app.add_exception_handler(ECommerceAPIError, ecommerce_exception_handler)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router Registration
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(products_router)
app.include_router(orders_router)
app.include_router(payments_router)

@app.get("/", tags=["Health Check"])
def root_health_check():
    return {
        "status": "healthy",
        "app_name": "Secure E-Commerce Backend V4",
        "environment": settings.ENVIRONMENT
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
