# ==============================================================================
# Program    : FastAPI Main Application Entry Point (main.py)
# Objective  : FastAPI application factory, router registration, and exception handlers.
# Concept    : Composition Root & Application Lifecycle Management
# Why Used   : Assembles routers, database dependencies, and global error handling handlers.
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
from app.routers import auth_router, users_router, products_router, orders_router

app = FastAPI(
    title="Secure E-Commerce Backend V3 API",
    description="Production-grade REST API with JWT Authentication, Role-Based Access Control & Alembic Migrations.",
    version="3.0.0",
    debug=settings.DEBUG
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Custom Exception Handler
app.add_exception_handler(ECommerceAPIError, ecommerce_exception_handler)

# Include Application Routers
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(products_router)
app.include_router(orders_router)

@app.get("/", tags=["Health"])
def read_root():
    return {
        "status": "online",
        "message": "Welcome to Secure E-Commerce Backend V3 API with JWT Authentication & RBAC",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
