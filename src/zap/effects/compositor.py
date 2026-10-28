"""Stacks the active effects into the outgoing frame.

One presence effect is always installed (you: visible / dissolving / vanished /
reassembling). At most one LOOK, one CUTAWAY and one MODAL run at a time;
OVERLAYs stack. An effect that raises is aborted and the frame still goes out;
one that fails three times within a minute is disabled for the session.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque

import numpy as np

from zap.effects.base import Effect, FxContext, Layer

log = logging.getLogger(__name__)

MAX_ERRORS = 3
ERROR_WINDOW_S = 60.0


class Compositor:
    def __init__(self, presence: Effect | None = None) -> None:
        self.presence = presence
        self.look: Effect | None = None
        self.cutaway: Effect | None = None
        self.modal: Effect | None = None
        self.overlays: list[Effect] = []
        self.disabled: set[str] = set()
        self._errors: dict[str, deque[float]] = defaultdict(deque)
        self._out: np.ndarray | None = None
        self.error_times: deque[float] = deque(maxlen=32)  # every effect failure, for the supervisor

    # ------------------------------------------------------------ control
    def active(self) -> list[Effect]:
        slots = [self.presence, self.look, self.cutaway, self.modal, *self.overlays]
        return [e for e in slots if e is not None and e.active]

    def busy_with_look(self) -> bool:
        return any(e is not None and e.active for e in (self.look, self.cutaway, self.modal))

    def needs_mask(self) -> bool:
        return any(e.needs_mask for e in self.active())

    def needs_face(self) -> bool:
        return any(e.needs_face for e in self.active())

    def start(self, effect: Effect, ctx: FxContext, **kw) -> bool:
        if effect.name in self.disabled:
            return False
        if effect.layer == Layer.PRESENCE:
            raise ValueError("presence is installed once, not started")
        if effect.layer == Layer.OVERLAY:
            self.overlays = [e for e in self.overlays if e.name != effect.name and e.active]
            self.overlays.append(effect)
        else:
            slot = {Layer.LOOK: "look", Layer.CUTAWAY: "cutaway", Layer.MODAL: "modal"}[effect.layer]
            previous = getattr(self, slot)
            if previous is not None and previous is not effect and previous.active:
                previous.abort()
            setattr(self, slot, effect)
        try:
            effect.start(ctx, **kw)
        except Exception:
            log.exception("effect %s failed to start", effect.name)
            self._failed(effect, ctx.now)
            return False
        return True

    def release(self, name: str, ctx: FxContext) -> None:
        for e in self.active():
            if e.name == name:
                e.release(ctx)

    def find(self, name: str) -> Effect | None:
        for e in self.active():
            if e.name == name:
                return e
        return None

    def clear(self) -> None:
        """PANIC: drop every effect immediately and bring the user back."""
        for e in (self.look, self.cutaway, self.modal, *self.overlays):
            if e is not None:
                e.abort()
        self.look = self.cutaway = self.modal = None
        self.overlays = []
        if self.presence is not None and hasattr(self.presence, "reset"):
            self.presence.reset()

    # ------------------------------------------------------------ render
    def render(self, ctx: FxContext) -> np.ndarray:
        if self._out is None or self._out.shape != ctx.frame.shape:
            self._out = np.empty_like(ctx.frame)
        out = self._out
        np.copyto(out, ctx.frame)
        for e in (self.presence, self.look, *self.overlays, self.cutaway, self.modal):
            if e is None or not e.active:
                continue
            try:
                e.update(ctx, out)
            except Exception:
                log.exception("effect %s crashed; aborting it", e.name)
                self._failed(e, ctx.now)
                if e is self.presence and hasattr(e, "reset"):
                    e.reset()
                    np.copyto(out, ctx.frame)
        self.overlays = [e for e in self.overlays if e.active]
        return out

    def _failed(self, effect: Effect, now: float) -> None:
        self.error_times.append(now)
        if effect is not self.presence:
            effect.abort()
        q = self._errors[effect.name]
        q.append(now)
        while q and now - q[0] > ERROR_WINDOW_S:
            q.popleft()
        if len(q) >= MAX_ERRORS and effect is not self.presence:
            self.disabled.add(effect.name)
            log.error("effect %s disabled for this session after %d errors", effect.name, len(q))
