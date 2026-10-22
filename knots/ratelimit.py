"""
A small in-process rate limit, because the endpoint is public and every call can spend Bedrock, TMDB and YouTube
quota. Fixed window per key, in memory: it protects one server process (one EC2 instance), which is the deployment
this ships with. It is not a distributed limiter and it resets on restart.

  KNOTS_RATE_LIMIT        calls allowed per key per window (default 30); 0 turns it off
  KNOTS_RATE_WINDOW_S     window length in seconds (default 60)
  KNOTS_RATE_GLOBAL       calls allowed per window across everyone (default 300); 0 turns it off
"""
from __future__ import annotations

import os
import threading
import time

MAX_KEYS = 10000            # bounded memory: an attacker cannot grow this by inventing keys


class RateLimiter:
    def __init__(self, per_key: int | None = None, window_s: float | None = None, overall: int | None = None):
        self.per_key = int(os.environ.get("KNOTS_RATE_LIMIT", "30")) if per_key is None else per_key
        self.window_s = float(os.environ.get("KNOTS_RATE_WINDOW_S", "60")) if window_s is None else window_s
        self.overall = int(os.environ.get("KNOTS_RATE_GLOBAL", "300")) if overall is None else overall
        self._lock = threading.Lock()
        self._counts: dict[str, tuple[float, int]] = {}
        self._all: tuple[float, int] = (0.0, 0)

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            if self.overall > 0:
                start, n = self._all
                if now - start >= self.window_s:
                    start, n = now, 0
                if n >= self.overall:
                    return False
                self._all = (start, n + 1)
            if self.per_key <= 0:
                return True
            start, n = self._counts.get(key, (now, 0))
            if now - start >= self.window_s:
                start, n = now, 0
            if n >= self.per_key:
                if self.overall > 0:                               # a refused call does not use up the shared budget
                    self._all = (self._all[0], max(0, self._all[1] - 1))
                return False
            if len(self._counts) >= MAX_KEYS and key not in self._counts:
                self._counts = {k: v for k, v in self._counts.items() if now - v[0] < self.window_s}
                if len(self._counts) >= MAX_KEYS:
                    self._counts.clear()
            self._counts[key] = (start, n + 1)
            return True
