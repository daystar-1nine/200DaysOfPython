# ==============================================================================
# Program    : FastAPI Composition Root & Global Exception Handlers (main.py)
# Objective  : Configure logging, ASGI middleware, global exception handlers, OpenAPI metadata, and route handlers.
# Concept    : Production-Ready FastAPI Composition Root
# Why Used   : Wires logging, middleware, global error handlers, and routers into a production-grade application.
# ==============================================================================

import os
import sys
import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.config import settings
from app.logging_config import setup_logging
from app.exceptions import ECommerceAPIError
from app.middleware.request_id import RequestIDMiddleware
from app.middleware.timing import TimingMiddleware
from app.routers import (
    auth_router,
    users_router,
    products_router,
    orders_router,
    payments_router,
    health_router
)

# Initialize application logging infrastructure
setup_logging()
logger = logging.getLogger("app.main")

app = FastAPI(
    title="Production-Ready E-Commerce Backend V5",
    description="Enterprise-grade FastAPI backend featuring structured logging, custom exception handling, OpenAPI docs, middleware, and health probes.",
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Register ASGI Observability Middleware (Request ID & Timing)
app.add_middleware(TimingMiddleware)
app.add_middleware(RequestIDMiddleware)

# Register CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# GLOBAL EXCEPTION HANDLERS (Standardized Error JSON Structure)
# ------------------------------------------------------------------------------

@app.exception_handler(ECommerceAPIError)
async def domain_exception_handler(request: Request, exc: ECommerceAPIError) -> JSONResponse:
    """Global Exception Handler converting domain custom exceptions to standardized JSON error structure."""
    request_id = getattr(request.state, "request_id", "N/A")
    logger.warning(
        f"Domain exception intercepted: code='{exc.code}', message='{exc.message}'",
        extra={"request_id": request_id}
    )

    headers = {}
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        headers["WWW-Authenticate"] = "Bearer"
    if request_id != "N/A":
        headers["X-Request-ID"] = request_id

    error_payload = {
        "error": {
            "code": exc.code,
            "message": exc.message,
            "request_id": request_id,
            "fields": None
        }
    }
    return JSONResponse(status_code=exc.status_code, content=error_payload, headers=headers)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Global Exception Handler formatting Pydantic validation errors into standardized JSON structure."""
    request_id = getattr(request.state, "request_id", "N/A")
    logger.warning(
        f"Request validation failed: {str(exc)}",
        extra={"request_id": request_id}
    )

    field_errors = {}
    for err in exc.errors():
        field = ".".join([str(loc) for loc in err["loc"] if loc != "body"])
        field_errors[field or "request"] = err["msg"]

    headers = {}
    if request_id != "N/A":
        headers["X-Request-ID"] = request_id

    error_payload = {
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed. Please check field inputs.",
            "request_id": request_id,
            "fields": field_errors
        }
    }
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=error_payload, headers=headers)

# ------------------------------------------------------------------------------
# ROUTER REGISTRATION
# ------------------------------------------------------------------------------
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(products_router)
app.include_router(orders_router)
app.include_router(payments_router)

@app.get("/", tags=["Health & Observability"])
def root_index():
    return {
        "message": "Welcome to Production-Ready E-Commerce API V5",
        "documentation": "/docs",
        "redoc": "/redoc",
        "liveness": "/health",
        "readiness": "/health/ready"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
