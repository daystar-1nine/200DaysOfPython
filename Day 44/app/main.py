# ==============================================================================
# Program    : User Management API V3 Composition Root (main.py)
# Objective  : Create database tables automatically via Base.metadata.create_all(), register routers & exception handlers.
# Concept    : Composition Root & Database Table Auto-Creation
# Why Used   : Initializes database schemas and starts ASGI web server.
# ==============================================================================

import os
import sys
from fastapi import FastAPI

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import engine, Base
from app.exceptions import UserAPIError, user_api_exception_handler
from app.routers.users import router as users_router

# What is used : Base.metadata.create_all(bind=engine)
# Why it is used: Automatically creates 'users' database table on startup if it does not exist
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="User Management API V3",
    description="Database-Backed REST API built with FastAPI, SQLAlchemy 2.0, PostgreSQL/SQLite, and Pydantic.",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Register Custom Exception Handler
app.add_exception_handler(UserAPIError, user_api_exception_handler)

# Include Routers
app.include_router(users_router)

@app.get("/", tags=["General"])
def read_root():
    return {
        "message": "Welcome to User Management API V3 (Database Persistent)",
        "database": "SQLAlchemy 2.0 ORM + PostgreSQL/SQLite",
        "documentation": "/docs"
    }

@app.get("/health", tags=["General"])
def health_check():
    return {"status": "healthy", "database": "connected"}

@app.get("/about", tags=["General"])
def about_api():
    return {
        "application": "User Management API V3",
        "version": "3.0.0",
        "framework": "FastAPI",
        "orm": "SQLAlchemy 2.0",
        "author": "Suraj Sawant",
        "repository": "https://github.com/daystar-1nine/200DaysOfPython"
    }
