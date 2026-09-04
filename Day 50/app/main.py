"""
===============================================================================
DAY 50 — TASKFLOW API COMPOSITION ROOT (MAIN APPLICATION)
===============================================================================
This module constructs the FastAPI application instance, registers custom ASGI
middleware, mounts domain routers, and sets up global exception handlers.
===============================================================================
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.config import settings
from app.database import Base, engine
from app.exceptions import TaskFlowException
from app.middleware.request_id import RequestIDMiddleware
from app.middleware.timing import TimingMiddleware
from app.routers import auth_router, users_router, tasks_router, admin_router, health_router
from app.logging_config import logger

# What is used: Base.metadata.create_all for SQLite fallback environment table initialization.
# Why it is used: Ensures tables exist when running in local development mode.
# How it works: Compiles ORM models and creates database tables if not present.
Base.metadata.create_all(bind=engine)

# What is used: FastAPI application initialization with OpenAPI metadata.
# Why it is used: Configures API root metadata, Swagger docs path (/docs), and title.
# How it works: Instantiates FastAPI app instance.
app = FastAPI(
    title="TaskFlow API",
    description="Production-style Task Management REST API for Day 50 Milestone",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# What is used: ASGI Middleware registration.
# Why it is used: Attaches X-Request-ID correlation tracking and Process-Time-Ms timing headers.
# How it works: Passes middleware classes to app.add_middleware.
app.add_middleware(RequestIDMiddleware)
app.add_middleware(TimingMiddleware)

# What is used: APIRouter mounting.
# Why it is used: Combines endpoint modules into a unified application router tree.
# How it works: Registers auth, users, tasks, admin, and health routers.
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(tasks_router)
app.include_router(admin_router)
app.include_router(health_router)


@app.exception_handler(TaskFlowException)
async def taskflow_exception_handler(request: Request, exc: TaskFlowException) -> JSONResponse:
    """Global exception handler for custom TaskFlow domain exceptions."""
    # What is used: Custom exception interception and standardized JSON error formatting.
    # Why it is used: Converts domain exceptions into standardized ErrorPayload response structures.
    # How it works: Extracts code, message, and request_id state, returning JSONResponse with status_code.
    request_id = getattr(request.state, "request_id", "N/A")
    logger.warning({"event": "domain_exception", "code": exc.code, "message": exc.message, "request_id": request_id})
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "request_id": request_id,
                "details": exc.details,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Global exception handler for Pydantic request validation errors."""
    # What is used: Pydantic RequestValidationError handler.
    # Why it is used: Intercepts request validation errors and formats standardized 422 JSON error.
    # How it works: Extracts validation error details array and returns standardized JSON payload.
    request_id = getattr(request.state, "request_id", "N/A")
    logger.warning({"event": "validation_error", "details": exc.errors(), "request_id": request_id})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "UNPROCESSABLE_ENTITY",
                "message": "Input validation failed.",
                "request_id": request_id,
                "details": {"fields": exc.errors()},
            }
        },
    )


@app.exception_handler(Exception)
async def global_unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global fallback exception handler for unexpected 500 server errors."""
    # What is used: Catch-all exception interceptor.
    # Why it is used: Prevents raw unhandled stack trace leakage to API clients.
    # How it works: Logs error exception traceback and returns 500 Internal Server Error payload.
    request_id = getattr(request.state, "request_id", "N/A")
    logger.error({"event": "unhandled_exception", "error": str(exc), "request_id": request_id}, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred. Please contact support.",
                "request_id": request_id,
            }
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
