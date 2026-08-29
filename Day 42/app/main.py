# ==============================================================================
# Program    : User Management FastAPI Main Application (main.py)
# Objective  : Instantiate FastAPI app, register routes, configure exception handlers, and expose /health and /about endpoints.
# Concept    : Composition Root & Application Entry Point
# Why Used   : Creates central ASGI web application instance for Uvicorn server execution.
# ==============================================================================

import os
import sys
from fastapi import FastAPI, status

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.exceptions import UserAPIError, user_api_exception_handler
from app.routes.users import router as users_router

app = FastAPI(
    title="User Management REST API",
    description="Production-style REST API built with FastAPI, Pydantic, and Uvicorn.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Register Custom Exception Handler
app.add_exception_handler(UserAPIError, user_api_exception_handler)

# Include API Routers
app.include_router(users_router)

# What is used : GET / Root Endpoint
# Why it is used: Returns welcome message and links to interactive docs
@app.get("/", tags=["General"])
def read_root():
    return {
        "message": "Welcome to the User Management REST API",
        "documentation": "/docs",
        "redoc": "/redoc"
    }

# What is used : GET /health Bonus Endpoint
# Why it is used: Exposes system health status for monitoring & load balancers
@app.get("/health", tags=["General"])
def health_check():
    return {"status": "healthy"}

# What is used : GET /about Bonus Endpoint
# Why it is used: Provides application metadata and environment info
@app.get("/about", tags=["General"])
def about_api():
    return {
        "application": "User Management REST API",
        "version": "1.0.0",
        "framework": "FastAPI",
        "author": "Suraj Sawant",
        "repository": "https://github.com/daystar-1nine/200DaysOfPython"
    }
