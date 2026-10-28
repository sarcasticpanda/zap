"""Time-based hysteresis for held gestures.

A pose counts only after it has been held (mostly) continuously for ``t_on``,
ends only after it has been gone for ``t_off``, and cannot fire again until a
cooldown has passed *and* the user has visibly let go. Everything is measured
in seconds, never frames, so a slow camera doesn't make gestures harder::

    IDLE -> ARMING -> ACTIVE -> COOLDOWN -> WAIT_RELEASE -> IDLE

Landmark models drop a hand for a frame or two all the time, so short runs of
false samples (< ``dropout``) don't reset ARMING; instead the time-weighted
share of true samples over the last ``t_on`` seconds must reach ``ratio``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

IDLE, ARMING, ACTIVE, COOLDOWN, WAIT_RELEASE = "idle", "arming", "active", "cooldown", "wait_release"
STATES = (IDLE, ARMING, ACTIVE, COOLDOWN, WAIT_RELEASE)

_EPS = 1e-6  # float slack so 12 steps of 1/30 s count as 0.4 s


@dataclass
class DwellParams:
    t_on: float  # hold this long before "on"
    t_off: float  # gone this long before "off"
    min_show: float  # "off" never comes sooner than this after "on"
    cooldown: float  # after "off", ignore the pose this long
    rearm: float = 0.4  # then require it released this long
    ratio: float = 0.8  # time-weighted share of true samples needed over the arming window
    dropout: float = 0.12  # false runs shorter than this don't abort arming


class Dwell:
    """One gesture's hold/release state machine. Call ``update`` once per frame."""

    def __init__(self, params: DwellParams, name: str = "") -> None:
        self.params = params
        self.name = name
        self.state = IDLE
        self.progress = 0.0  # 0..1 while ARMING, 1 while ACTIVE (for the HUD)
        self._entered = 0.0
        self._last_t: float | None = None
        self._false_since: float | None = None  # start of the current run of false samples
        self._seg: deque[list] = deque()  # [t0, t1, truth] spans inside the arming window
        self._true_s = 0.0
        self._total_s = 0.0

    # ------------------------------------------------------------ control
    def reset(self) -> None:
        """Forget everything: back to IDLE (a still-held pose re-arms on the next frame)."""
        self._enter(IDLE, self._last_t or 0.0)
        self._false_since = None

    def cancel(self, now: float) -> None:
        """Drop this activation without a cooldown; the pose must be released before it can arm again."""
        self._enter(WAIT_RELEASE, now)

    def update(self, truth: bool, now: float, t_on: float | None = None) -> str | None:
        """Feed one sample. Returns "on", "off" or None. ``t_on`` overrides the hold time for this frame."""
        p = self.params
        t_on = p.t_on if t_on is None else max(0.0, t_on)
        prev_t = now if self._last_t is None else min(self._last_t, now)
        self._last_t = now
        if truth:
            self._false_since = None
        elif self._false_since is None:
            self._false_since = now
        false_for = 0.0 if truth else now - self._false_since

        if self.state == IDLE and truth:
            self._enter(ARMING, now)
            prev_t = now
        if self.state == ARMING:
            return self._arming(truth, now, prev_t, false_for, t_on)
        if self.state == ACTIVE:
            if not truth and false_for + _EPS >= p.t_off and now - self._entered + _EPS >= p.min_show:
                self._enter(COOLDOWN, now)
                return "off"
            return None
        if self.state == COOLDOWN and now - self._entered + _EPS >= p.cooldown:
            self._enter(WAIT_RELEASE, now)
        if self.state == WAIT_RELEASE and not truth and false_for + _EPS >= p.rearm:
            self._enter(IDLE, now)
        return None

    # ------------------------------------------------------------ queries
    def ratio(self) -> float:
        """Time-weighted share of true samples in the current arming window."""
        return 1.0 if self._total_s <= _EPS else self._true_s / self._total_s

    def cooldown_left(self, now: float) -> float:
        if self.state != COOLDOWN:
            return 0.0
        return max(0.0, self.params.cooldown - (now - self._entered))

    def info(self, now: float) -> dict[str, Any]:
        return {"state": self.state, "progress": round(self.progress, 3), "cooldown_left": round(self.cooldown_left(now), 2)}

    # ------------------------------------------------------------ internals
    def _arming(self, truth: bool, now: float, prev_t: float, false_for: float, t_on: float) -> str | None:
        p = self.params
        if not truth and false_for + _EPS >= p.dropout:
            self._enter(IDLE, now)
            return None
        self._push(prev_t, now, truth)
        self._trim(now - t_on)
        elapsed = now - self._entered
        self.progress = 1.0 if t_on <= 0 else min(1.0, elapsed / t_on)
        if truth and elapsed + _EPS >= t_on and self.ratio() + _EPS >= p.ratio:
            self._enter(ACTIVE, now)
            return "on"
        return None

    def _enter(self, state: str, now: float) -> None:
        self.state = state
        self._entered = now
        self.progress = 1.0 if state == ACTIVE else 0.0
        self._seg.clear()
        self._true_s = self._total_s = 0.0

    def _push(self, t0: float, t1: float, truth: bool) -> None:
        if t1 <= t0:
            return
        self._seg.append([t0, t1, truth])
        self._total_s += t1 - t0
        if truth:
            self._true_s += t1 - t0

    def _trim(self, start: float) -> None:
        seg = self._seg
        while seg and seg[0][1] <= start:
            t0, t1, truth = seg.popleft()
            self._total_s -= t1 - t0
            if truth:
                self._true_s -= t1 - t0
        if seg and seg[0][0] < start:
            cut = start - seg[0][0]
            seg[0][0] = start
            self._total_s -= cut
            if seg[0][2]:
                self._true_s -= cut
        if not seg:
            self._true_s = self._total_s = 0.0
