# ==============================================================================
# Program    : Request ID Correlation Middleware (request_id.py)
# Objective  : Generate or preserve unique X-Request-ID header for distributed request tracing.
# Concept    : Distributed Request Tracing & Correlation ID
# Why Used   : Correlates millions of log entries down to a single HTTP request lifecycle.
# ==============================================================================

import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class RequestIDMiddleware(BaseHTTPMiddleware):
    """ASGI Middleware assigning unique X-Request-ID to request state and response headers."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Preserve client-supplied X-Request-ID or generate new UUID hex string
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = uuid.uuid4().hex[:12]

        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
