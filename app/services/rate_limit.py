from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock
import time
from typing import Deque

from fastapi import HTTPException, Request, status


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, *, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        now = time.time()
        earliest = now - window_seconds
        retry_after = 1
        with self._lock:
            bucket = self._buckets[key]
            while bucket and bucket[0] < earliest:
                bucket.popleft()
            if len(bucket) >= limit:
                retry_after = max(1, int(bucket[0] + window_seconds - now))
                return False, retry_after
            bucket.append(now)
        return True, retry_after

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()


_limiter = InMemoryRateLimiter()


def _request_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def enforce_rate_limit(
    *,
    request: Request,
    scope: str,
    limit: int,
    window_seconds: int,
    identity: str | None = None,
) -> None:
    id_part = (identity or "").strip().lower()
    key = f"{scope}|{_request_ip(request)}|{id_part}"
    allowed, retry_after = _limiter.check(key=key, limit=limit, window_seconds=window_seconds)
    if allowed:
        return
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=f"Too many requests for {scope}. Try again in {retry_after} second(s).",
        headers={"Retry-After": str(retry_after)},
    )


def clear_rate_limiter() -> None:
    _limiter.clear()
