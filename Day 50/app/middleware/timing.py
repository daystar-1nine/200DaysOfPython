"""
===============================================================================
DAY 50 — REQUEST EXECUTION TIMING MIDDLEWARE
===============================================================================
This middleware measures request execution duration and appends a Process-Time-Ms
header to outgoing HTTP responses for latency monitoring.
===============================================================================
"""

import time
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class TimingMiddleware(BaseHTTPMiddleware):
    """ASGI Middleware recording request lifecycle latency."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # What is used: Monotonic high-precision timer (time.perf_counter).
        # Why it is used: Measures exact elapsed execution duration in milliseconds.
        # How it works: Calculates elapsed time and attaches Process-Time-Ms response header.
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["Process-Time-Ms"] = f"{process_time_ms:.2f}"
        return response
