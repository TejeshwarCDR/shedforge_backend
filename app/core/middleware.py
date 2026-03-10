from __future__ import annotations

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import Settings


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, settings: Settings) -> None:
        super().__init__(app)
        self._settings = settings

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
        if self._settings.security_enable_hsts:
            max_age = max(1, self._settings.security_hsts_max_age_seconds)
            response.headers.setdefault("Strict-Transport-Security", f"max-age={max_age}; includeSubDomains")
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, max_bytes: int) -> None:
        super().__init__(app)
        self._max_bytes = max(1, max_bytes)

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.scope.get("type") != "http":
            return await call_next(request)

        raw_length = request.headers.get("content-length")
        if raw_length:
            try:
                value = int(raw_length)
            except ValueError:
                value = 0
            if value > self._max_bytes:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": (
                            f"Request body too large ({value} bytes). "
                            f"Maximum allowed is {self._max_bytes} bytes."
                        )
                    },
                )
        return await call_next(request)
