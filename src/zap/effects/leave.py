"""Leaving the meeting (snap twice): a countdown after you vanish, then ZAP leaves Zoom.

The meeting already sees the empty room (the Thanos vanish), so this modal
draws nothing; your preview shows the countdown. Leaving is the one
irreversible thing ZAP does, so the countdown is cancellable right up to
zero, ``on_finish`` fires exactly once and only when the countdown completes,
and in "thumbs_up" mode a missing confirmation cancels rather than leaves.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from zap.effects.base import Effect, FxContext, Layer
from zap.effects.registry import register

log = logging.getLogger(__name__)

CONFIRMING, COUNTING, LEAVING, CANCELLED, DONE = "confirming", "counting", "leaving", "cancelled", "done"


@register("leave")
class Leave(Effect):
    """Count down, then ``on_finish()`` once.

    ``confirm``: "countdown" starts counting at once; "thumbs_up" first waits up
    to ``confirm_timeout_s`` for :meth:`confirm` and cancels on timeout.
    ``on_finish`` runs on the render thread inside ``update``; it must only
    hand the leave off to a worker thread and return.
    """

    layer = Layer.MODAL

    def __init__(
        self,
        countdown_s: float = 3.0,
        confirm: str = "countdown",
        on_finish: Callable[[], Any] | None = None,
        confirm_timeout_s: float = 5.0,
        hold_s: float = 10.0,
        **params: Any,
    ) -> None:
        if confirm not in ("countdown", "thumbs_up"):
            raise ValueError(f"confirm must be 'countdown' or 'thumbs_up', got {confirm!r}")
        super().__init__(countdown_s=countdown_s, confirm=confirm, on_finish=on_finish,
                         confirm_timeout_s=confirm_timeout_s, hold_s=hold_s, **params)
        self.countdown_s = max(0.1, float(countdown_s))
        self.confirm_mode = confirm
        self.on_finish = on_finish
        self.confirm_timeout_s = float(confirm_timeout_s)
        self.hold_s = float(hold_s)
        self._state = DONE
        self._finished = False  # on_finish has been called
        self._t_count = 0.0
        self._t_finish = 0.0
        self._confirm_pending = False
        self._seconds_left = self.countdown_s

    # ------------------------------------------------------------ HUD
    @property
    def state(self) -> str:
        """"confirming" | "counting" | "leaving" | "cancelled" | "done"."""
        return self._state

    @property
    def seconds_left(self) -> float:
        """Countdown seconds remaining (full while confirming, 0 once leaving)."""
        return self._seconds_left

    @property
    def finished(self) -> bool:
        """True once on_finish has been called."""
        return self._finished

    # ------------------------------------------------------------ control
    def start(self, ctx: FxContext, **kw: Any) -> None:
        self.active = True
        self.t0 = ctx.now
        self._finished = False
        self._confirm_pending = False
        self._seconds_left = self.countdown_s
        if self.confirm_mode == "thumbs_up":
            self._state = CONFIRMING
        else:
            self._state, self._t_count = COUNTING, ctx.now

    def confirm(self, ctx: FxContext | None = None) -> bool:
        """Thumbs-up received. Returns False if there was nothing to confirm."""
        if not self.active or self._state != CONFIRMING:
            return False
        if ctx is not None:
            self._state, self._t_count = COUNTING, ctx.now
        else:
            self._confirm_pending = True
        return True

    def cancel(self, ctx: FxContext | None = None) -> bool:
        """Stop now. Returns True if this prevented on_finish (too late once leaving)."""
        if not self.active or self._state in (CANCELLED, DONE, LEAVING):
            return False
        self._state = CANCELLED
        self._confirm_pending = False
        self._seconds_left = 0.0
        self.active = False
        return not self._finished

    def abort(self) -> None:
        self.active = False
        if self._state != CANCELLED:
            self._state = DONE

    # ------------------------------------------------------------ per frame
    def update(self, ctx: FxContext, out: np.ndarray) -> None:
        now = ctx.now
        if self._state == CONFIRMING:
            if self._confirm_pending:
                self._confirm_pending = False
                self._state, self._t_count = COUNTING, now
            elif now - self.t0 >= self.confirm_timeout_s:
                log.info("leave: no thumbs-up within %.1f s; cancelled", self.confirm_timeout_s)
                self.cancel(ctx)
                return
            else:
                return
        if self._state == COUNTING:
            tc = now - self._t_count
            self._seconds_left = max(0.0, self.countdown_s - tc)
            if tc >= self.countdown_s:
                self._state, self._t_finish = LEAVING, now
                self._fire()
        if self._state == LEAVING:
            self._seconds_left = 0.0
            if now - self._t_finish >= self.hold_s:
                self.active, self._state = False, DONE

    def _fire(self) -> None:
        self._finished = True  # set first: on_finish must never run twice
        if self.on_finish is None:
            log.info("leave countdown finished (no on_finish handler)")
            return
        try:
            self.on_finish()
        except Exception:
            log.exception("leave on_finish handler failed")
