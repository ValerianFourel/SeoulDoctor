"""Small, bounded in-memory rate limiter for public API boundaries."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from math import ceil
from threading import Lock
from time import monotonic
from typing import Deque, Optional


@dataclass(frozen=True)
class RateLimitDecision:
    """Result of consuming one request from a sliding window."""

    allowed: bool
    remaining: int
    retry_after_seconds: int


class SlidingWindowRateLimiter:
    """Thread-safe per-key limiter with bounded client-key memory."""

    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
        max_clients: int = 10_000,
    ) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be positive")
        if window_seconds < 1:
            raise ValueError("window_seconds must be positive")
        if max_clients < 1:
            raise ValueError("max_clients must be positive")

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_clients = max_clients
        self._requests: OrderedDict[str, Deque[float]] = OrderedDict()
        self._lock = Lock()

    def check(self, key: str, now: Optional[float] = None) -> RateLimitDecision:
        """Consume one request and return whether the window permits it."""
        current_time = monotonic() if now is None else float(now)
        cutoff = current_time - self.window_seconds

        with self._lock:
            requests = self._requests.get(key)
            if requests is None:
                if len(self._requests) >= self.max_clients:
                    self._requests.popitem(last=False)
                requests = deque()
                self._requests[key] = requests
            else:
                self._requests.move_to_end(key)

            while requests and requests[0] <= cutoff:
                requests.popleft()

            if len(requests) >= self.max_requests:
                retry_after = max(
                    1,
                    ceil(requests[0] + self.window_seconds - current_time),
                )
                return RateLimitDecision(
                    allowed=False,
                    remaining=0,
                    retry_after_seconds=retry_after,
                )

            requests.append(current_time)
            return RateLimitDecision(
                allowed=True,
                remaining=self.max_requests - len(requests),
                retry_after_seconds=0,
            )
