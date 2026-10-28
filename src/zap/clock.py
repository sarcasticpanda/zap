"""Time sources. Everything in ZAP is timed in seconds from ``perf_counter``,
never in frame counts, so behaviour survives fps drops. Replays swap in a
``ReplayClock`` so the same recording always produces the same events."""

from __future__ import annotations

import time


class Clock:
    def now(self) -> float:
        return time.perf_counter()


class ReplayClock(Clock):
    def __init__(self, t: float = 0.0) -> None:
        self._t = t

    def set(self, t: float) -> None:
        self._t = t

    def now(self) -> float:
        return self._t
