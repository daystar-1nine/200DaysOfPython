# ==============================================================================
# Program    : User Management API V2 Composition Root (main.py)
# Objective  : Instantiate FastAPI app, register modular routers, add exception handlers, and expose /health, /about, /config endpoints.
# Concept    : Clean Composition Root & Multi-Router Inclusion
# Why Used   : Keeps main.py small and clean by delegating route definitions to routers.
# ==============================================================================

import os
import sys
from fastapi import FastAPI, Depends

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.config import get_settings
from app.exceptions import UserAPIError, user_api_exception_handler
from app.routers.users import router as users_router
from app.routers.products import router as products_router

app = FastAPI(
    title="User Management API V2",
    description="Layered Architecture & Dependency Injection API built with FastAPI.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Exception Handler Registration
app.add_exception_handler(UserAPIError, user_api_exception_handler)

# Include Routers
app.include_router(users_router)
app.include_router(products_router)

@app.get("/", tags=["General"])
def read_root():
    return {
        "message": "Welcome to User Management API V2",
        "architecture": "Router -> Dependency -> Service -> Repository -> Data",
        "documentation": "/docs"
    }

@app.get("/config", tags=["General"])
def read_config(settings: dict = Depends(get_settings)):
    return settings

@app.get("/health", tags=["General"])
def health_check():
    return {"status": "healthy"}

@app.get("/about", tags=["General"])
def about_api():
    return {
        "application": "User Management API V2",
        "version": "2.0.0",
        "framework": "FastAPI",
        "author": "Suraj Sawant",
        "repository": "https://github.com/daystar-1nine/200DaysOfPython"
    }
