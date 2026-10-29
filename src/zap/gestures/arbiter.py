"""Decides which gestures actually do something.

Detectors only say "this frame looks like a pose" or "that was two snaps". The
arbiter turns that into at most a few Intents per frame. It applies hold times
and cooldowns (one Dwell per pose), arming and the grace period after arming,
priorities between gestures that overlap, and the rules that stop effects
piling up: one pose effect at a time, a short gap between different ones, and
nothing new while the user is vanished or the leave countdown (modal) runs.

Keyboard commands are deliberate, so they skip the gesture gates. A modal can
always be cancelled, armed or not: stopping an accidental "leave meeting" is a
safety feature.

Priorities: panic 100 > modal cancel/confirm 90 > snaps 80 > two-hand poses 60
> one-hand signs 55 > hand+face poses 50 > face-only poses 20.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any

from zap.clock import Clock
from zap.config import Config
from zap.gestures.dwell import ACTIVE, ARMING, Dwell
from zap.gestures.poses import ARBITER, ArbiterParams, load_params
from zap.gestures.types import (
    ALL_GESTURES,
    CANCEL,
    COLLAR,
    COVER_FACE,
    EYEBROWS_UP,
    FOREHEAD,
    HAND_SIGNS,
    HANDS_UP,
    L_SHAPE,
    PANIC,
    PEACE,
    SNAP1,
    SNAP2,
    SNAP3,
    THUMBS_UP,
    TWO_THUMBS,
    Intent,
    SnapBurst,
)

log = logging.getLogger(__name__)

VISIBLE, DISSOLVING, VANISHED, REASSEMBLING = "visible", "dissolving", "vanished", "reassembling"

PRIORITY: dict[str, int] = {
    SNAP1: 80, SNAP2: 80, SNAP3: 80,
    HANDS_UP: 60, COVER_FACE: 60, TWO_THUMBS: 55, PEACE: 55, L_SHAPE: 55,
    FOREHEAD: 50, COLLAR: 50,
    EYEBROWS_UP: 20,
}
_KIND_PRIORITY = {PANIC: 100, "cancel": 90, "confirm": 90, "charge": 80, "discharge": 80}

TRIGGER_POSES = (HANDS_UP, COVER_FACE, TWO_THUMBS, *HAND_SIGNS, FOREHEAD, COLLAR, EYEBROWS_UP)  # by priority
OVERLAY_POSES = (TWO_THUMBS,)  # their effect plays on top of a meme that's still showing
MODAL_POSES = (CANCEL, THUMBS_UP)  # only listened to while the modal runs (or, the palm, while a card shows)
HAND_POSES = (HANDS_UP, COVER_FACE, TWO_THUMBS, *HAND_SIGNS, FOREHEAD, COLLAR)  # silenced by a loaded snap
CARD_GRACE_S = 1.0  # a card ignores the palm this long: the hand that just made the sign opens as it drops
EVENT_POSES = (COVER_FACE,)  # one-frame truths: the candidate need not still be held when it fires
HELD_EFFECTS = (HANDS_UP,)  # the dwell's "off" releases the effect
_SNAP_BY_COUNT = {1: SNAP1, 2: SNAP2, 3: SNAP3}
MODAL_GRACE_S = 1.5  # leaving ignores cancels this long: the open hand that just snapped is not a "cancel"
_EPS = 1e-6


def _priority(intent: Intent) -> int:
    return _KIND_PRIORITY.get(intent.kind, PRIORITY.get(intent.gesture, 0))


def _classify(ev: Any) -> tuple[str | None, tuple[float, float] | None]:
    """Snap-stream item -> ("burst" | "charge" | "discharge" | None, where)."""
    if isinstance(ev, SnapBurst):
        return "burst", ev.where
    kind, where = ev, None
    if isinstance(ev, tuple) and ev:
        kind, where = ev[0], (ev[1] if len(ev) > 1 else None)
    elif hasattr(ev, "kind"):
        kind, where = ev.kind, getattr(ev, "where", None)
    return (kind, where) if kind in ("charge", "discharge") else (None, None)


class Arbiter:
    """Gesture arbitration. The app sets ``presence_state`` and ``modal_active`` every frame,
    then calls ``update``; hotkey commands go through ``command``.
    """

    def __init__(self, cfg: Config, clock: Clock | None = None) -> None:
        self.clock = clock or Clock()
        self.armed = cfg.app.armed_on_start
        self.arm_time = self.clock.now()
        self.presence_state = VISIBLE
        self.modal_active = False
        self.card_active = False  # a freeze card (BRB / TBC) shows: an open palm ends it
        self.busy = False
        self.suppressed: dict[str, str] = {}
        self._seen: dict[str, bool] = {}  # pose truths last frame (for the "pose seen" log)
        self.last_intents: deque[Intent] = deque(maxlen=8)
        self._manual: set[str] = set()  # held effects started by a hotkey (released by the next press)
        self.configure(cfg)

    def configure(self, cfg: Config) -> None:
        """(Re)load settings, e.g. after a config reload. Resets every gesture in progress."""
        self.cfg = cfg
        self.grace = cfg.app.arm_grace_s
        self.params = load_params(cfg.gestures)
        self.p: ArbiterParams = self.params[ARBITER]
        self.dwells = {g: Dwell(self.params[g].dwell(), g) for g in (*TRIGGER_POSES, *MODAL_POSES)}
        self._reset_gestures()

    def _reset_gestures(self) -> None:
        for d in self.dwells.values():
            d.reset()
        self._pending: dict[str, float] = {}  # pose -> time its dwell said "on"
        self._last_true: dict[str, float] = {}  # pose -> last time its gated truth was True
        self._holding: set[str] = set()  # held effects started by a gesture (released by the dwell)
        self._last_pose = ("", -math.inf)
        self._busy_t = -math.inf
        self._loaded_t = -math.inf
        self._charge_t: float | None = None  # a snap burst may be starting
        self._charged = False  # a "charge" intent went out with no burst or discharge yet
        self._was_modal = False
        self._modal_t = -1e9
        self._was_card = False
        self._card_t = -1e9

    # ------------------------------------------------------------ commands
    def command(self, name: str, now: float | None = None) -> list[Intent]:
        """Apply a hotkey command; returns the intents to act on now. Unknown names return []."""
        now = self.clock.now() if now is None else now
        if name == PANIC:
            # Gestures stay armed (a disarmed app looked "broken"); they pause for the arm grace.
            log.warning("PANIC: stopping every effect; gestures pause for %.1f s", self.grace)
            self._reset_gestures()
            self.arm_time = now
            self._manual.clear()
            out = [Intent(PANIC, "", now)]
        elif name in ("arm_toggle", "arm", "disarm"):
            out = self._set_armed(not self.armed if name == "arm_toggle" else name == "arm", now)
        elif name == "confirm":
            out = [Intent("confirm", THUMBS_UP, now)] if self.modal_active else []
            if not out:
                log.info("confirm ignored: nothing to confirm")
        elif name.startswith("trigger:"):
            out = self._manual_trigger(name.split(":", 1)[1], now)
        else:
            return []
        self.last_intents.extend(out)
        return out

    def _set_armed(self, armed: bool, now: float) -> list[Intent]:
        out = []
        if armed and not self.armed:
            self._reset_gestures()
            self.arm_time = now
            log.info("armed (gestures live in %.1f s)", self.grace)
        elif not armed and self.armed:
            out = [Intent("release", g, now) for g in sorted(self._holding)]
            if self._charged:
                out.append(Intent("discharge", "", now))
            self._reset_gestures()
            log.info("disarmed")
        self.armed = armed
        return out

    def _manual_trigger(self, gesture: str, now: float) -> list[Intent]:
        if gesture not in ALL_GESTURES:
            log.warning("unknown gesture in command 'trigger:%s'", gesture)
            return []
        if gesture in HELD_EFFECTS:  # a hotkey can't be "held": a second press releases
            if gesture in self._manual or gesture in self._holding:
                self._manual.discard(gesture)
                self._holding.discard(gesture)
                self._pending.pop(gesture, None)
                self.dwells[gesture].cancel(now)
                return [Intent("release", gesture, now)]
            self._manual.add(gesture)
        return [Intent("trigger", gesture, now)]

    # ------------------------------------------------------------ per frame
    def update(
        self,
        pose_truths: Mapping[str, bool],
        snap_events: Iterable[Any] = (),
        now: float | None = None,
        *,
        busy: bool = False,
        snap_loaded: Any = False,
        t_on: Mapping[str, float] | None = None,
    ) -> list[Intent]:
        """One frame. ``pose_truths`` from PoseDetector.update; ``snap_events`` are SnapBursts
        and "charge"/"discharge" markers (str, (kind, where) tuple, or anything with ``.kind``);
        ``busy`` = a look/cutaway effect is showing; ``snap_loaded`` truthy = a hand is in a
        loaded snap pinch; ``t_on`` = per-gesture hold-time overrides (PoseDetector.t_on).
        Returns intents, highest priority first.
        """
        now = self.clock.now() if now is None else now
        self.busy = busy
        if busy:
            self._busy_t = now
        if len(snap_loaded) > 0 if isinstance(snap_loaded, (list, tuple)) else bool(snap_loaded):
            self._loaded_t = now
        self.suppressed = {}
        if self.modal_active and not self._was_modal:
            self._modal_t = now
            for g in MODAL_POSES:  # a palm already open when the modal starts must be lowered first
                self.dwells[g].cancel(now)
        self._was_modal = self.modal_active
        if self.card_active and not self._was_card:
            self._card_t = now
            self.dwells[CANCEL].cancel(now)  # likewise for a card
        self._was_card = self.card_active

        out: list[Intent] = []
        self._snaps(snap_events, now, out)
        self._modal_poses(pose_truths, now, out)
        self._poses(pose_truths, now, t_on or {}, out)
        out.sort(key=_priority, reverse=True)
        for i in out:
            log.info("intent %s %s", i.kind, i.gesture)
        self.last_intents.extend(out)
        return out

    def _live_block(self, now: float) -> str | None:
        """Why no gesture may fire at all right now, or None."""
        if not self.armed:
            return "disarmed"
        if now - self.arm_time < self.grace:
            return "grace"
        return None

    # ------------------------------------------------------------ snaps
    def _snaps(self, events: Iterable[Any], now: float, out: list[Intent]) -> None:
        if self._charge_t is not None and now - self._charge_t > self.p.charge_timeout_s:
            self._charge_t = None
        for ev in events:
            kind, where = _classify(ev)
            if kind == "burst":
                self._charge_t = None
                self._burst(ev, now, out)
            elif kind == "charge":
                self._charge_t = now
                ok = self._live_block(now) is None and not self.modal_active
                if ok and self.presence_state in (VISIBLE, VANISHED):
                    self._charged = True
                    out.append(Intent("charge", "", now, where))
            elif kind == "discharge":
                self._charge_t = None
                self._discharge(now, out, where)
            else:
                log.warning("ignoring unknown snap event %r", ev)

    def _burst(self, b: SnapBurst, now: float, out: list[Intent]) -> None:
        g = _SNAP_BY_COUNT.get(b.count)
        if g is None:
            log.info("ignoring a burst of %d snaps", b.count)
            self._discharge(now, out, b.where)
            return
        if self.modal_active:  # any snap cancels the modal, armed or not ...
            self._discharge(now, out, b.where)
            if now - self._modal_t >= MODAL_GRACE_S:  # ... but not the hand still finishing the snaps
                out.append(Intent("cancel", g, now, b.where))
            return
        reason = self._live_block(now)
        guard_left = self.cfg.zoom.leave_guard_s - (now - self.arm_time)
        if reason is None and g == SNAP2 and guard_left > 0:
            reason = "leave guard"
            log.info("snap x2 ignored: leave guard for another %.0f s after arming", guard_left)
        if reason is not None:
            self.suppressed[g] = reason
            self._discharge(now, out, b.where)
            return
        self._charged = False  # the effect takes over from the pre-roll
        out.append(Intent("trigger", g, now, b.where))

    def _discharge(self, now: float, out: list[Intent], where: tuple[float, float] | None = None) -> None:
        if self._charged:
            self._charged = False
            out.append(Intent("discharge", "", now, where))

    # ------------------------------------------------------------ poses
    def _modal_poses(self, truths: Mapping[str, bool], now: float, out: list[Intent]) -> None:
        for g in MODAL_POSES:
            raw = bool(truths.get(g, False))
            reason = self._modal_block(g, now)
            if raw and reason and (self.modal_active or self.card_active):
                self.suppressed[g] = reason
            if self.dwells[g].update(raw and reason is None, now) == "on":
                out.append(Intent("cancel" if g == CANCEL else "confirm", g, now))

    def _modal_block(self, g: str, now: float) -> str | None:
        if g == CANCEL and self.card_active and not self.modal_active:  # an open palm ends BRB / TBC
            return "card just started" if now - self._card_t < CARD_GRACE_S else None
        if not self.modal_active:
            return "no modal"
        if g == CANCEL and not self.cfg.zoom.palm_cancels:
            return "open palm doesn't cancel"
        if now - self._modal_t < MODAL_GRACE_S:
            return "leaving just started"
        if g == THUMBS_UP:
            if self.cfg.zoom.confirm != "thumbs_up":
                return "confirm mode is countdown"
            return self._live_block(now)
        return None

    def _pose_block(self, g: str, now: float) -> str | None:
        """Why pose ``g`` must not even arm right now, or None."""
        reason = self._live_block(now)
        if reason:
            return reason
        if self.modal_active:
            return "modal"
        if self.presence_state != VISIBLE:
            return self.presence_state
        if g in HAND_POSES and now - self._loaded_t < self.p.snap_hold_s:
            return "snap loaded"
        if g in (FOREHEAD, COLLAR) and self.dwells[HANDS_UP].state in (ARMING, ACTIVE):
            return "hands up"
        return None

    def _poses(self, truths: Mapping[str, bool], now: float, t_on: Mapping[str, float], out: list[Intent]) -> None:
        for g in TRIGGER_POSES:  # priority order, so hands_up's state is current for forehead/collar
            raw = bool(truths.get(g, False))
            reason = self._pose_block(g, now)
            if raw and not self._seen.get(g):
                log.info("pose seen: %s%s", g, f" (blocked: {reason})" if reason else "")
            self._seen[g] = raw
            if raw and reason:
                self.suppressed[g] = reason
            gated = raw and reason is None
            if gated:
                self._last_true[g] = now
            d = self.dwells[g]
            ev = d.update(gated, now, t_on.get(g))
            if ev == "on":
                self._pending[g] = now
            elif ev == "off":
                if g in self._holding:
                    self._holding.discard(g)
                    out.append(Intent("release", g, now))
                elif g in self._pending and g not in EVENT_POSES:
                    del self._pending[g]  # let go before it could fire: no cooldown for nothing shown
                    d.reset()
        self._fire(now, out)

    def _fire(self, now: float, out: list[Intent]) -> None:
        for g, t0 in list(self._pending.items()):
            if g not in EVENT_POSES and self.dwells[g].state != ACTIVE:
                del self._pending[g]
            elif now - t0 > self.p.pending_max_s:
                log.info("%s dropped: could not fire (%s)", g, self._fire_block(g, now))
                self._drop(g, now)
        if not self._pending or any(i.kind == "trigger" for i in out):
            return  # nothing waiting, or a snap already won this frame
        g = min(self._pending, key=lambda k: (-PRIORITY[k], self._pending[k]))
        reason = self._fire_block(g, now)
        if reason:
            self.suppressed.setdefault(g, reason)
            return
        del self._pending[g]
        out.append(Intent("trigger", g, now))
        self._last_pose = (g, now)
        if g in HELD_EFFECTS:
            self._holding.add(g)
        for other in list(self._pending):  # one look/cutaway at a time
            self._drop(other, now)

    def _fire_block(self, g: str, now: float) -> str | None:
        """Why the pending pose ``g`` can't fire this frame, or None."""
        p = self.p
        reason = self._pose_block(g, now)
        if reason:
            return reason
        if now - self._pending[g] + _EPS < p.settle_s:
            return "settling"
        if g not in EVENT_POSES and now - self._last_true.get(g, -math.inf) > self.dwells[g].params.dropout:
            return "not held"
        for h in TRIGGER_POSES:
            if PRIORITY[h] > PRIORITY[g] and self.dwells[h].state == ARMING:
                return f"{h} arming"
        if self._charge_t is not None:
            return "snap arming"
        if g in OVERLAY_POSES:  # money rain can fall on top of a meme
            return None
        if self.busy:
            return "busy"
        last_g, last_t = self._last_pose
        if now - self._busy_t < p.gap_s or (last_g != g and now - last_t < p.gap_s):
            return "gap"
        return None

    def _drop(self, g: str, now: float) -> None:
        self._pending.pop(g, None)
        if g in EVENT_POSES:
            self.dwells[g].reset()  # the moment passed; the user may simply try again
        else:
            self.dwells[g].cancel(now)  # still held: it must be let go before it can fire again

    # ------------------------------------------------------------ HUD
    def status(self, now: float | None = None) -> dict[str, Any]:
        now = self.clock.now() if now is None else now
        since = now - self.arm_time
        return {
            "armed": self.armed,
            "grace_left": round(max(0.0, self.grace - since), 2) if self.armed else 0.0,
            "leave_guard_left": round(max(0.0, self.cfg.zoom.leave_guard_s - since), 1) if self.armed else 0.0,
            "presence": self.presence_state,
            "modal": self.modal_active,
            "card": self.card_active,
            "busy": self.busy,
            "snap_arming": self._charge_t is not None,
            "dwell": {g: d.info(now) for g, d in self.dwells.items()},
            "pending": sorted(self._pending),
            "holding": sorted(self._holding | self._manual),
            "suppressed": dict(self.suppressed),
            "last": [{"t": round(i.t, 3), "kind": i.kind, "gesture": i.gesture} for i in self.last_intents],
        }
