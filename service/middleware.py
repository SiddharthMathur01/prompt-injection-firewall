"""
service/middleware.py

FastAPI middleware for the firewall service.

Provides:
  - Request timing and logging
  - Source identity extraction from headers/API keys
  - Rate limiting per source
  - Error handling
"""
import logging
import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("pif.middleware")


class TimingMiddleware(BaseHTTPMiddleware):
    """Add X-Process-Time header to all responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.time()
        response = await call_next(request)
        process_time = (time.time() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
        return response


class SourceIdentityMiddleware(BaseHTTPMiddleware):
    """
    Extract source identity from the request for behavioral tracking.
    
    Checks (in order):
    1. X-Source-ID header
    2. Authorization header (hashed)
    3. Client IP
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        source_id = request.headers.get("X-Source-ID")
        
        if not source_id:
            auth = request.headers.get("Authorization", "")
            if auth:
                import hashlib
                source_id = f"key-{hashlib.sha256(auth.encode()).hexdigest()[:12]}"
            else:
                source_id = request.client.host if request.client else "unknown"
        
        request.state.source_id = source_id
        response = await call_next(request)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple per-source rate limiter.
    
    Limits requests per source per minute to prevent abuse.
    This is a basic in-memory implementation — use Redis-backed
    rate limiting for production multi-worker deployments.
    """
    
    def __init__(self, app, max_requests_per_minute: int = 60):
        super().__init__(app)
        self.max_rpm = max_requests_per_minute
        self._buckets: dict[str, list[float]] = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        source_id = getattr(request.state, "source_id", "unknown")
        now = time.time()
        cutoff = now - 60
        
        # Prune old entries
        self._buckets[source_id] = [
            t for t in self._buckets[source_id] if t > cutoff
        ]
        
        if len(self._buckets[source_id]) >= self.max_rpm:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit",
                        "code": "too_many_requests",
                    }
                },
            )
        
        self._buckets[source_id].append(now)
        response = await call_next(request)
        return response
