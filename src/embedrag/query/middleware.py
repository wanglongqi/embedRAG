"""FastAPI middleware for request observability on the query node.

``RequestContextMiddleware`` injects a unique request ID into structlog
context, logs request start/end with timing and status code, and attaches
the request ID to the response ``X-Request-ID`` header for distributed
tracing correlation.
"""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from embedrag.logging_setup import get_logger

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Injects request_id, logs request start/end with timing."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        structlog.contextvars.bind_contextvars(request_id=request_id)

        t0 = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request_error", method=request.method, path=request.url.path)
            raise
        finally:
            elapsed_ms = round((time.monotonic() - t0) * 1000, 2)
            logger.info(
                "request",
                method=request.method,
                path=request.url.path,
                status=getattr(response, "status_code", 500) if "response" in dir() else 500,
                elapsed_ms=elapsed_ms,
            )
            structlog.contextvars.unbind_contextvars("request_id")

        response.headers["X-Request-ID"] = request_id
        return response
