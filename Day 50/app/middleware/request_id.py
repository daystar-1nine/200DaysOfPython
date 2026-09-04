"""
===============================================================================
DAY 50 — REQUEST ID TRACING MIDDLEWARE
===============================================================================
This middleware ensures every HTTP request has a unique X-Request-ID header
for distributed request context tracing.
===============================================================================
"""

import uuid
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """ASGI Middleware injecting X-Request-ID headers into requests and responses."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # What is used: Header extraction and UUID4 generation fallback.
        # Why it is used: Assigns unique correlation ID to every incoming request context.
        # How it works: Reuses existing X-Request-ID header or generates uuid4 hex string.
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
