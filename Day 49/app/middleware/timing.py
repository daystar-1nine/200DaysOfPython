# ==============================================================================
# Program    : Request Timing Execution Latency Middleware (timing.py)
# Objective  : Measure request execution duration and attach Process-Time-Ms header.
# Concept    : Latency Observability Middleware
# Why Used   : Identifies slow endpoints and logs performance statistics in production.
# ==============================================================================

import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("app.middleware.timing")

class TimingMiddleware(BaseHTTPMiddleware):
    """ASGI Middleware measuring endpoint processing duration in milliseconds."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()
        request_id = getattr(request.state, "request_id", "N/A")

        response: Response = await call_next(request)

        process_time_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["Process-Time-Ms"] = f"{process_time_ms:.2f}"

        logger.info(
            f"{request.method} {request.url.path} -> status={response.status_code} duration={process_time_ms:.2f}ms",
            extra={"request_id": request_id}
        )

        return response
