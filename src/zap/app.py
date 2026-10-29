"""ZAP's frame loop: webcam -> MediaPipe -> gestures -> meme effects -> OBS Virtual Camera.

Two stages run side by side (a pipeline): a worker thread runs the MediaPipe
models on frame N while the main thread decides gestures, composites and sends
frame N-1. Perception is the slow part (~25 ms for the hand model alone), so
doing both one after the other could not reach 30 fps on a laptop. Each frame
travels together with the Signals computed from it, so a person mask always
cuts out the picture it was computed on; if the main thread falls behind, the
older result is dropped instead of queued.

The main thread owns all app state: hotkeys are queued by their own thread and
applied here, and the Zoom "leave" runs on a worker so it never stalls video.
A supervisor trims work when the laptop can't keep up (L1) and falls back to
a plain passthrough if it really struggles (L2), recovering once healthy.
"""

from __future__ import annotations

import ctypes
import faulthandler
import json
import logging
import logging.handlers
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from zap import config as zap_config
from zap import paths
from zap.assets import AssetResolver
from zap.clock import Clock
from zap.config import Config, ConfigError
from zap.effects import enhance, faceswap, registry
from zap.effects.base import FxContext, Layer, NullSfx
from zap.effects.compositor import Compositor
from zap.effects.plate import BackgroundPlate
from zap.gestures import calibrate
from zap.gestures.arbiter import Arbiter
from zap.gestures.calibrate import Calibration
from zap.gestures.dwell import ARMING
from zap.gestures.poses import PoseDetector
from zap.gestures.types import (
    ALL_GESTURES,
    CANCEL,
    COLLAR,
    COVER_FACE,
    EYEBROWS_UP,
    FOREHEAD,
    HAND_SIGNS,
    HANDS_UP,
    PANIC,
    SNAP1,
    SNAP2,
    SNAP3,
    Intent,
)
from zap.signals import FramePacket, Signals
from zap.vision.perception import Timings

log = logging.getLogger(__name__)

PLATE_FILE = "plate.png"
PLATE_COUNTDOWN_S = 2.0  # time to step out of view after Ctrl+Alt+B
PLATE_CAPTURE_S = 3.0
PLATE_MAX_PERSON = 0.02  # share of the frame the person mask may still cover for a "clean" capture
PLATE_EVERY_S = 0.1  # feed the plate at most this often (it costs 3-5 ms)
STATS_EVERY_S = 5.0
DUMP_EVERY_S = 2.0  # ZAP_PREVIEW_DUMP
FACE_CHIN_IDX = 152
LOW_CHIN = 0.78  # chin below this share of the frame height: the collar is out of the picture
NOTICE_S = 8.0
STALL_CARD_EVERY_S = 0.2
FACE_POSES = (HANDS_UP, FOREHEAD, COVER_FACE, EYEBROWS_UP, COLLAR, *HAND_SIGNS)  # signs: is the hand off the face?
CARDS = ("brb", "tbc")  # freeze cards: an open palm (or their hotkey again) ends them
MAIN_STAGES = ("gestures", "render", "send", "preview", "loop")
EXIT_OK, EXIT_CRASH, EXIT_SETUP, EXIT_RUNNING = 0, 1, 2, 3


class ModelsMissing(RuntimeError):
    pass


class AlreadyRunning(RuntimeError):
    pass


# ---------------------------------------------------------------- small helpers
class Notices:
    """HUD messages that expire; safe to add from any thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, tuple[float, str, str]] = {}  # key -> (expires, severity, text)
        self._n = 0

    def add(self, text: str, severity: str = "info", seconds: float = NOTICE_S, key: str | None = None) -> None:
        with self._lock:
            self._n += 1
            self._items[key or f"_{self._n}"] = (time.perf_counter() + seconds, severity, text)
        getattr(log, {"error": "error", "warn": "warning"}.get(severity, "info"))("HUD: %s", text)

    def clear(self, key: str) -> None:
        with self._lock:
            self._items.pop(key, None)

    def active(self) -> list[tuple[str, str]]:
        now = time.perf_counter()
        with self._lock:
            for k in [k for k, (exp, _, _) in self._items.items() if exp < now]:
                del self._items[k]
            return [(sev, text) for _, sev, text in self._items.values()]


def setup_logging(verbose: bool = False) -> None:
    """Log to logs/zap.log (rotating) and the console; dump native crashes to logs/fault.log."""
    paths.logs_dir().mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if any(getattr(h, "_zap", False) for h in root.handlers):
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    fh = logging.handlers.RotatingFileHandler(paths.logs_dir() / "zap.log", maxBytes=2_000_000, backupCount=3,
                                              encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    for h in (fh, ch):
        h._zap = True  # type: ignore[attr-defined]
        root.addHandler(h)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    try:
        fault = open(paths.logs_dir() / "fault.log", "a", encoding="utf-8")  # noqa: SIM115 - lives as long as the process
        faulthandler.enable(fault)
    except OSError:
        log.warning("could not open fault.log", exc_info=True)


def acquire_single_instance(name: str = "Local\\ZAP-gesture-camera") -> Any:
    """Hold a named mutex so a second ZAP can't fight over the camera; raise AlreadyRunning if taken."""
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.restype = wintypes.HANDLE
    k32.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
    handle = k32.CreateMutexW(None, False, name)
    if handle and ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        k32.CloseHandle(handle)
        raise AlreadyRunning("ZAP is already running (look for its preview window or the tray of another terminal).")
    return handle


def release_single_instance(handle: Any) -> None:
    if handle and sys.platform == "win32":
        ctypes.WinDLL("kernel32").CloseHandle(handle)


# ---------------------------------------------------------------- pipeline stage 1
class PerceptionWorker:
    """Newest camera frame -> Perception.run, on its own thread; the main thread takes the newest result."""

    def __init__(self, source: Any, perception: Any) -> None:
        self.source = source  # anything with wait_latest(timeout) -> FramePacket | None
        self.perception = perception
        self.passthrough = False  # degrade level 2: skip the models entirely
        self._needs = (False, False, False)
        self._cond = threading.Condition()
        self._item: tuple[FramePacket, Signals] | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="zap-perception-worker", daemon=True)
        self.processed = 0
        self.dropped = 0

    def start(self) -> PerceptionWorker:
        self._thread.start()
        return self

    def set_needs(self, face: bool, mask: bool, hands: bool) -> None:
        self._needs = (face, mask, hands)

    def get(self, timeout: float = 0.5) -> tuple[FramePacket, Signals] | None:
        with self._cond:
            if self._item is None:
                self._cond.wait(timeout)
            item, self._item = self._item, None
            return item

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            pkt = self.source.wait_latest(0.5)
            if pkt is None:
                continue
            h, w = pkt.frame.shape[:2]
            face, mask, hands = self._needs
            if self.passthrough or self.perception is None:
                sig = Signals(pkt.t, pkt.seq, width=w, height=h)
            else:
                try:
                    sig = self.perception.run(pkt, need_face=face, need_mask=mask, need_hands=hands)
                except Exception:  # Perception already contains model errors; this is a last resort
                    log.exception("perception failed on a frame")
                    sig = Signals(pkt.t, pkt.seq, width=w, height=h)
            with self._cond:
                if self._item is not None:
                    self.dropped += 1
                self._item = (pkt, sig)
                self.processed += 1
                self._cond.notify_all()


# ---------------------------------------------------------------- snaps
LAST_FACE_S = 8.0  # effects may fall back to a face seen this recently (a hand can hide it)
QUIET_S = 0.5  # camera snaps: a burst ends this soon after the last snap once the hand is lowered


class SnapInput:
    """Snap detection -> SnapBursts and charge/discharge markers for the arbiter.

    Snaps are seen by the camera (``snap.vision_only``, the default: works in a
    noisy room); with ``vision_only = false`` the mic must also hear them.
    """

    def __init__(self, cfg: Config, use_mic: bool = True) -> None:
        from zap.gestures.snap import BurstCounter, SnapFusion
        from zap.gestures.snap_audio import SnapAudioDetector

        s = cfg.snap
        self.gap_s = s.burst_gap_ms / 1000.0
        self.det = SnapAudioDetector(rate=48000, sensitivity=s.sensitivity)
        self.mic = None
        ok = False
        if use_mic and not s.vision_only:
            from zap.devices.audio_in import MicStream

            self.mic = MicStream(self.det, device=s.mic or None)
            ok = self.mic.start()
        self.chose_vision = s.vision_only or not use_mic
        self.fusion = SnapFusion(av_offset_s=s.av_offset_ms / 1000.0, vision_only=not ok, burst_gap_s=self.gap_s)
        self.fusion.sensitivity = s.sensitivity
        if os.environ.get("ZAP_SNAP_TRACE"):  # per-frame hand features for tuning snap detection
            self.fusion.trace = open(paths.logs_dir() / "snap_trace.jsonl", "w", encoding="utf-8", buffering=1)  # noqa: SIM115
        self.bursts = BurstCounter(gap_s=self.gap_s, quiet_s=QUIET_S,
                                   max_count=2 if cfg.bindings.get("snap3", "none") == "none" else 3)
        self.recent: deque[tuple[bool, str]] = deque(maxlen=5)

    @property
    def vision_only(self) -> bool:
        return self.fusion.vision_only

    @property
    def pending(self) -> bool:
        """A snap may be in progress: run the hand model on every frame."""
        mic = self.mic.pending_onset if self.mic is not None else None
        return mic is not None or self.fusion.pending_onset is not None or self.bursts.in_burst             or self.fusion.watching

    def _vision_only(self) -> None:
        self.fusion.vision_only = True

    def step(self, sig: Signals, now: float, gate_until: float = 0.0) -> list[Any]:
        mic = self.mic
        if mic is not None and mic.error and not self.fusion.vision_only:
            log.warning("microphone failed (%s); snaps are now seen by the camera only", mic.error)
            self._vision_only()
        if mic is not None and not mic.error:
            if gate_until:
                mic.gate(gate_until)  # our own sound effects are not snaps
            for c in mic.drain():
                self.recent.append((c.accepted, c.reason))
                self.fusion.add_candidate(c)
        self.fusion.add_frame(sig)
        events: list[Any] = []
        for t, where in self.fusion.resolve(now, in_burst=self.bursts.in_burst):
            if self.fusion.vision_only:
                self.recent.append((True, "seen"))
                log.info("snap seen")
            marker = self.bursts.add(t, where)
            if marker is not None:
                events.append(marker)
        hold = not self.fusion.vision_only or self.fusion.hand_up  # hand lowered: no more snaps coming
        done, markers = self.bursts.tick(now, mic.pending_onset if mic is not None else None,
                                         self.fusion.pending_onset, hold=hold)
        return events + done + markers

    def close(self) -> None:
        if self.mic is not None:
            self.mic.stop()
        if self.fusion.trace is not None:
            self.fusion.trace.close()


# ---------------------------------------------------------------- dispatch
class EffectController:
    """Turns arbiter Intents into effects, through ``cfg.bindings``."""

    def __init__(self, cfg: Config, compositor: Compositor, presence: Any, notices: Notices | None = None,
                 leave: Callable[..., Any] | None = None, force_dry_run: bool = False) -> None:
        self._instant: str | None = None  # presence state before a first snap acted instantly
        self.cfg = cfg
        self.compositor = compositor
        self.presence = presence
        self.notices = notices or Notices()
        self._leave = leave
        self.force_dry_run = force_dry_run
        self.leave_results: list[Any] = []
        self.leave_threads: list[threading.Thread] = []
        self._charge_where: tuple[float, float] | None = None

    # ------------------------------------------------------------ queries
    def target_for(self, gesture: str) -> str:
        return self.cfg.bindings.get(gesture, "none") or "none"

    @property
    def leaving(self) -> Any:
        """The leave-meeting countdown, while it runs."""
        m = self.compositor.modal
        return m if m is not None and m.active and m.name == "leave" else None

    @property
    def modal_active(self) -> bool:
        lv = self.leaving
        return lv is not None and lv.state in ("confirming", "counting")

    @property
    def card(self) -> Any:
        """The freeze card (We'll Be Right Back / To Be Continued) showing, if any."""
        c = self.compositor.cutaway
        return c if c is not None and c.active and c.name in CARDS else None

    @property
    def card_active(self) -> bool:
        c = self.card
        return c is not None and c.state != "out"

    # ------------------------------------------------------------ intents
    def handle(self, intent: Intent, ctx: FxContext) -> None:
        kind = intent.kind
        if kind == "trigger":
            self.trigger(intent.gesture, ctx, intent.where)
        elif kind == "release":
            self.release(intent.gesture, ctx)
        elif kind == "charge":
            self.charge(ctx, intent.where)
        elif kind == "discharge":
            self.discharge()
        elif kind == "cancel":
            if intent.gesture == CANCEL and not self.modal_active:
                self.finish_card(ctx)  # an open palm ends We'll Be Right Back / To Be Continued
            else:
                self.cancel_leave(ctx)
        elif kind == "confirm":
            lv = self.leaving
            if lv is not None and lv.confirm(ctx):
                self.notices.add("leave confirmed", key="leave")
        elif kind == PANIC:
            self.panic(ctx)

    def trigger(self, gesture: str, ctx: FxContext, where: tuple[float, float] | None = None) -> bool:
        if gesture in (SNAP1, SNAP2, SNAP3):
            if self._instant is not None:  # the first snap already toggled you
                prev, self._instant = self._instant, None
                if gesture == SNAP1:
                    return True
                if self.target_for(gesture) != "leave":  # leaving keeps you vanished (see _start_leave)
                    self._undo_toggle(prev)  # two snaps meant something else: back as you were, instantly
            self.discharge()  # the effect takes over from the ember pre-roll
        target = self.target_for(gesture)
        if target == "none":
            log.info("%s is not bound to anything", gesture)
            return False
        log.info("%s -> %s", gesture, target)
        return self.start_target(target, ctx, where)

    def start_target(self, target: str, ctx: FxContext, where: tuple[float, float] | None = None) -> bool:
        if target == "presence":
            ok = self.presence.toggle(ctx, where)
            if not ok:
                self.notices.add("can't vanish yet: no person mask (stand in view a moment)", "warn", key="presence")
            return ok
        if target == "leave":
            return self._start_leave(ctx, where)
        if getattr(self.presence, "hidden", False):
            self.presence.reappear(ctx, where)  # e.g. thumbs up while vanished: come back, then make it rain
        running = self.compositor.find(target) if target in CARDS else None
        if running is not None:  # the card's hotkey again: back to live
            running.finish(ctx)
            return True
        try:
            fx = self._make(target)
        except (KeyError, TypeError, ValueError) as e:
            self.notices.add(f"can't start '{target}': {e}", "error", key=f"fx:{target}")
            return False
        return self.compositor.start(fx, ctx)

    def _make(self, target: str) -> Any:
        if target.startswith("cutaway:"):
            name = target.split(":", 1)[1]
            return registry.create("cutaway", cutaway=name, **self.cfg.cutaways.get(name, {}))
        if target == "cinema":
            running = self.compositor.find("cinema")
            if running is not None:
                return running  # hands back up during the hold: Cinema.start keeps it going
            return registry.create("cinema", layout=self.cfg.cinema.layout, font=self.cfg.cinema.font)
        if target == "money":
            return registry.create("money", caption=self.cfg.money.caption, stamp_text=self.cfg.money.stamp_text)
        if target == "brb":
            b = self.cfg.brb
            return registry.create("brb", text=b.text, seconds=b.seconds, hold_when_away=b.hold_when_away)
        if target == "tbc":
            return registry.create("tbc", seconds=self.cfg.tbc.seconds)
        return registry.create(target)

    def finish_card(self, ctx: FxContext) -> bool:
        c = self.card
        if c is None:
            return False
        c.finish(ctx)
        return True

    def release(self, gesture: str, ctx: FxContext) -> None:
        name = self.target_for(gesture).split(":", 1)[0]
        self.compositor.release(name, ctx)

    def charge(self, ctx: FxContext, where: tuple[float, float] | None) -> None:
        if self.target_for(SNAP1) == "presence" and self._instant is None:
            # act on the first snap right away (waiting to see if a second follows feels laggy);
            # a second snap undoes this instantly and plays its own effect
            prev = self.presence.state
            if self.presence.toggle(ctx, where):
                self._instant = prev
                return
        fx = registry.create("charge")
        if self.compositor.start(fx, ctx, where=where):
            self._charge_where = where

    def _undo_toggle(self, prev: str) -> None:
        if prev == "vanished":
            self.presence.start_hidden()
        else:
            self.presence.reset()

    def discharge(self) -> None:
        if self._instant is not None:  # a burst that does nothing (or isn't allowed): undo the first snap
            self._undo_toggle(self._instant)
            self._instant = None
        fx = self.compositor.find("charge")
        if fx is not None:
            fx.discharge()
        self._charge_where = None

    def follow_charge(self, ctx: FxContext) -> None:
        """Keep the ember sparks on the snapping hand while the snap count is open."""
        fx = self.compositor.find("charge")
        if fx is None or self._charge_where is None or not ctx.sig.hands:
            return
        last = np.asarray(self._charge_where, np.float64)
        pcs = [h.palm_center for h in ctx.sig.hands]
        best = min(pcs, key=lambda p: float(np.hypot(*(p - last))))
        if float(np.hypot(*(best - last))) <= 0.25 * ctx.W:
            fx.follow((float(best[0]), float(best[1])))
            self._charge_where = (float(best[0]), float(best[1]))

    # ------------------------------------------------------------ leaving
    def _start_leave(self, ctx: FxContext, where: tuple[float, float] | None = None) -> bool:
        """Snap twice: vanish (unless already gone; a reappear in progress is reversed), then count down and leave."""
        if self.presence.state in ("visible", "reassembling") and not self.presence.vanish(ctx, where):
            self.notices.add("couldn't vanish (no person mask yet); leaving anyway", "warn", key="presence")
        z = self.cfg.zoom
        fx = registry.create("leave", countdown_s=z.countdown_s, confirm=z.confirm, on_finish=self._on_leave_finish)
        ok = self.compositor.start(fx, ctx)
        if ok:
            msg = "thumbs up to leave the meeting, snap or Esc to cancel" if z.confirm == "thumbs_up" else \
                f"leaving the meeting in {z.countdown_s:.0f} s: snap or Esc to cancel"
            self.notices.add(msg, "warn", z.countdown_s + 6, key="leave")
        return ok

    def _on_leave_finish(self) -> None:
        """Called once on the render thread when the countdown hits zero: hand the leave to a worker."""
        th = threading.Thread(target=self._leave_worker, name="zap-zoom-leave", daemon=True)
        self.leave_threads.append(th)
        th.start()

    def _leave_worker(self) -> None:
        z = self.cfg.zoom
        dry = self.force_dry_run or z.dry_run
        leave = self._leave
        if leave is None:
            from zap.devices.zoom_ctl import leave
        try:
            res = leave(end_for_all=z.end_for_all, dry_run=dry)
        except Exception as e:  # zoom_ctl.leave never raises, but a replacement might
            log.exception("leave failed")
            res = type("LeaveResult", (), {"ok": False, "message": str(e), "dry_run": dry})()
        self.leave_results.append(res)
        text = f"Zoom: {res.message}" + (" (set zoom.dry_run = false to really leave)" if res.dry_run else "")
        self.notices.add(text, "info" if res.ok else "error", 20, key="leave")

    def cancel_leave(self, ctx: FxContext | None = None) -> bool:
        """Stop the leave countdown and bring you back."""
        lv = self.leaving
        if lv is None or not lv.cancel(ctx):
            return False
        self.notices.add("leaving cancelled", key="leave")
        if ctx is not None and getattr(self.presence, "hidden", False):
            self.presence.reappear(ctx)
        return True

    def panic(self, ctx: FxContext) -> None:
        lv = self.leaving
        if lv is not None:
            lv.cancel(ctx)  # never leave after a panic
        self.compositor.clear()  # also brings a vanished user back
        ctx.sfx.stop_all()
        self._charge_where = None
        self.notices.add("PANIC: effects stopped; gestures back in a few seconds", "warn", 10,
                         key="panic")


# ---------------------------------------------------------------- supervisor
class Supervisor:
    """Degrade ladder. fps is judged relative to what the camera delivers (a 25 fps webcam is fine).

    L1 (render p95 > 20 ms or under 26 fps over 2 s): lighter perception and dissolve.
    L2 (under 15 fps for 3 s, or 3 effect errors within 60 s): plain passthrough, disarmed.
    One level back after ``recover_s`` healthy; the wait doubles each time L2 is re-entered.
    """

    WINDOW_S = 2.0

    def __init__(self, start: float, warmup_s: float = 6.0, recover_s: float = 10.0) -> None:
        self.level = 0
        self._ready_at = start + warmup_s
        self._recover = recover_s
        self._l2_recover = recover_s
        self._sends: deque[float] = deque()
        self._render: deque[tuple[float, float]] = deque()
        self._low_since: float | None = None
        self._healthy_since: float | None = None
        self._changed_at = start
        self.reason = ""

    def frame(self, now: float, render_ms: float) -> None:
        self._sends.append(now)
        self._render.append((now, render_ms))
        while self._sends and now - self._sends[0] > self.WINDOW_S:
            self._sends.popleft()
        while self._render and now - self._render[0][0] > self.WINDOW_S:
            self._render.popleft()

    def fps(self, now: float) -> float:
        n = sum(1 for t in self._sends if now - t <= self.WINDOW_S)
        return n / self.WINDOW_S

    def check(self, now: float, cam_fps: float, error_times: Any = ()) -> int | None:
        """Returns the new level when it changes, else None."""
        if now < self._ready_at or cam_fps <= 1.0:
            return None
        out = self.fps(now)
        rel = min(30.0, out * 30.0 / max(cam_fps, 1.0))
        render_p95 = float(np.percentile([ms for _, ms in self._render], 95)) if self._render else 0.0
        errors = sum(1 for t in error_times if t > max(now - 60.0, self._changed_at))
        self._low_since = (self._low_since or now) if rel < 15.0 else None
        new = self.level
        if errors >= 3 or (self._low_since is not None and now - self._low_since >= 3.0):
            new, self.reason = 2, ("effect errors" if errors >= 3 else f"only {out:.0f} fps")
        elif self.level == 0 and (render_p95 > 20.0 or rel < 26.0):
            new, self.reason = 1, f"render p95 {render_p95:.0f} ms, {out:.0f} fps"
        healthy = render_p95 <= 15.0 and rel >= 27.0 and errors == 0
        if new == self.level:
            if not healthy or self.level == 0:
                self._healthy_since = None
                return None
            self._healthy_since = self._healthy_since or now
            wait = self._l2_recover if self.level == 2 else self._recover
            if now - self._healthy_since < wait:
                return None
            new, self.reason = self.level - 1, "healthy again"
        if new == 2 and self.level < 2:
            self._l2_recover = min(80.0, self._l2_recover * 2) if self._changed_at > self._ready_at else self._recover
        self.level, self._changed_at, self._healthy_since = new, now, None
        return new


# ---------------------------------------------------------------- the app
@dataclass
class RunOptions:
    preview: bool | None = None  # None = config
    seconds: float | None = None  # auto-exit (smoke tests)
    armed: bool | None = None
    snap_entry: bool | None = None
    try_target: str | None = None  # "zap try": trigger this every try_every_s
    try_every_s: float = 4.0
    try_hold_s: float = 2.5  # held effects (cinema) are released after this
    use_mic: bool = True
    hotkeys: bool = True
    calibrate: str = "auto"  # "auto": calibrate on first use (no saved calibration yet); "now"; "off"
    quit_after_calibration: bool = False  # "zap calibrate": exit once the result has been shown


class App:
    """Everything wired together. ``run()`` opens the devices, loops, and always cleans up."""

    def __init__(self, cfg: Config, opts: RunOptions | None = None, clock: Clock | None = None) -> None:
        self.cfg = cfg
        self.opts = opts or RunOptions()
        self.clock = clock or Clock()
        if self.opts.armed:
            cfg.app.armed_on_start = True
        self.snap_entry = cfg.app.snap_entry if self.opts.snap_entry is None else self.opts.snap_entry
        if self.snap_entry:
            cfg.app.armed_on_start = True  # the snap that brings you in is a gesture
        self.W, self.H = cfg.output.width, cfg.output.height
        self.notices = Notices()
        self.stats = Timings(MAIN_STAGES, window=150)
        self.cam = self.vcam = self.perception = self.worker = self.snaps = self.sfx = None
        self.hotkeys = self.preview = None
        self._mutex = None
        self._quit = False
        self._deadline: float | None = None
        self._commands: deque[str] = deque()
        self._frames = 0
        self._first_frame_done = False
        self._last_face, self._last_face_t = None, -1e9
        self._plate_at: float | None = None
        self._plate_was_capturing = False
        self._plate_person = 0.0
        self._plate_clean = False
        self._plate_fed = -1.0
        self._last_stats = 0.0
        self._run_t0 = 0.0
        self._fps_hist: list[float] = []
        self._next_try = 0.0
        self._try_release_at: float | None = None
        self._last_card = 0.0
        self._last_sig: Signals | None = None
        self.assets = AssetResolver()
        self.plate = BackgroundPlate(self.W, self.H)
        registry.load_builtin()
        from zap.effects.thanos import Presence

        self.presence = Presence()
        self.compositor = Compositor(presence=self.presence)
        self.det = PoseDetector(cfg.gestures)
        saved = _load_calibration()
        self.det.apply_calibration(saved)
        self._calib: Calibration | None = None  # Ctrl+Alt+C steps in progress
        self._calib_job: dict[str, Any] | None = None  # ... then their analysis on a worker thread
        self._calib_shown = ""
        self._banner: tuple[str, str, float] | None = None  # big calibration prompt on the preview only
        self._banner_until = 0.0
        # first use: calibrate to this person before anything else (once their face is steadily in view)
        first = saved is None or saved.get("version") != calibrate.VERSION
        self._auto_calib = self.opts.calibrate == "now" or (self.opts.calibrate == "auto" and first)
        self._face_since: float | None = None
        self._quit_at: float | None = None
        self._dump_dir = os.environ.get("ZAP_PREVIEW_DUMP")  # dev aid: save the preview now and then
        self._dump_next = 0.0
        self._truths: dict[str, bool] = {}  # this frame's raw gesture truths (lit up in the preview)
        self._sig_dump = None  # dev aid: ZAP_SIGNALS_DUMP=<file.jsonl> logs landmarks + gesture verdicts per frame
        if os.environ.get("ZAP_SIGNALS_DUMP"):
            try:
                self._sig_dump = open(os.environ["ZAP_SIGNALS_DUMP"], "w", encoding="utf-8")  # noqa: SIM115
            except OSError:
                log.exception("could not open ZAP_SIGNALS_DUMP")
        self._show_marks = False  # M in the preview: draw the tracked points (always while calibrating)
        self.arbiter = Arbiter(cfg, self.clock)
        self.supervisor = Supervisor(self.clock.now())
        self.ctl = EffectController(cfg, self.compositor, self.presence, self.notices,
                                    force_dry_run=self.opts.try_target is not None)

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Open everything. Raises ModelsMissing, CameraError, VirtualCamError with friendly messages."""
        from zap import fetch_models
        from zap.devices.audio_out import SfxPlayer
        from zap.devices.camera import Camera
        from zap.devices.vcam import VirtualCam
        from zap.render import synth_sfx
        from zap.vision.perception import Perception

        paths.ensure_dirs()
        missing = fetch_models.missing()
        if missing:
            raise ModelsMissing(f"MediaPipe models missing ({', '.join(missing)}). Run: zap fetch-models")
        self._mutex = acquire_single_instance()
        threading.Thread(target=self._prewarm, name="zap-prewarm", daemon=True).start()
        log.info("loading MediaPipe models ...")
        self.perception = Perception(self.cfg.perception, self.W, self.H)
        self.vcam = VirtualCam(self.W, self.H, self.cfg.output.fps)
        self.cam = Camera.from_config(self.cfg).start()
        self.sfx = SfxPlayer(self.assets, synth_sfx.synth, enabled=self.cfg.sound.enabled,
                             output=self.cfg.sound.output, volume=self.cfg.sound.volume)
        self.snaps = SnapInput(self.cfg, use_mic=self.opts.use_mic)
        if self.opts.hotkeys:
            from zap.devices.hotkeys import Hotkeys

            self.hotkeys = Hotkeys()
            if not self.hotkeys.start():
                self.notices.add(self.hotkeys.error, "warn", 30, key="hotkeys")
        show = self.cfg.app.preview if self.opts.preview is None else self.opts.preview
        if show:
            from zap.ui.preview import PreviewThread as Preview

            self.preview = Preview(mirror=self.cfg.app.preview_mirror and not self.cfg.camera.mirror)
        if self.snap_entry:
            self.presence.start_hidden()
            log.info("snap entry: starting invisible; snap once (or Ctrl+Alt+1) to appear")
        now = self.clock.now()
        if self.arbiter.armed:
            self.arbiter.arm_time = now  # grace period and leave guard count from when video starts
        self.supervisor = Supervisor(now)
        if self.opts.seconds:
            self._deadline = now + self.opts.seconds
        self._run_t0 = self._last_stats = now
        self._next_try = self._run_t0 + 3.0
        self.worker = PerceptionWorker(self.cam, self.perception).start()

    def _prewarm(self) -> None:
        """Build sprites, maps and sounds off the frame loop so first triggers don't hitch."""
        from zap.effects import cinema, money
        from zap.effects.cards import BeRightBack, ToBeContinued
        from zap.effects.cutaway import Cutaway
        from zap.effects.sad import SadMode
        from zap.render import synth_sfx

        steps = (
            lambda: cinema.prewarm(self.W, self.H, self.cfg.cinema.font),
            lambda: money.prewarm(self.H, self.cfg.money.stamp_text, self.cfg.money.caption),
            lambda: ToBeContinued.prewarm(self.W, self.H),
            lambda: BeRightBack.prewarm(self.W, self.H, self.cfg.brb.text),
            lambda: Cutaway.prewarm(self.W, self.H, self.assets, self.cfg.cutaways),
            lambda: SadMode.prewarm(self.W, self.H, self.assets),
            synth_sfx.prewarm,
            # your own sound files: an MP3 takes a moment to decode, so not on the first trigger
            lambda: [self.assets.sound(k) for k in (*synth_sfx.KEYS, *(f"cutaway_{n}" for n in self.cfg.cutaways))],
        )
        for step in steps:
            try:
                step()
            except Exception:
                log.exception("prewarm step failed (the effect will build on first use)")

    def run(self) -> int:
        code = EXIT_OK
        try:
            self.start()
            while not self._quit:
                item = self.worker.get(timeout=0.5)
                if self._deadline is not None and self.clock.now() >= self._deadline:
                    log.info("--seconds reached; exiting")
                    break
                if item is None:
                    self._idle()
                    continue
                try:
                    self.step(*item)
                except Exception:
                    log.exception("frame failed; sending the plain camera frame")
                    self.compositor.error_times.append(self.clock.now())
                    self._send(item[0].frame)
        except KeyboardInterrupt:
            log.info("Ctrl+C: exiting")
        finally:
            self.close()
        return code

    def quit(self) -> None:
        self._quit = True

    def close(self) -> None:
        self._log_summary()
        for name, fn in (
            ("worker", lambda: self.worker and self.worker.stop()),
            ("camera", lambda: self.cam and self.cam.stop()),
            ("snaps", lambda: self.snaps and self.snaps.close()),
            ("sound", lambda: self.sfx and self.sfx.close()),
            ("hotkeys", lambda: self.hotkeys and self.hotkeys.stop()),
            ("preview", lambda: self.preview and self.preview.close()),
            ("virtual camera", lambda: self.vcam and self.vcam.close()),
            ("perception", lambda: self.perception and self.perception.close()),
        ):
            try:
                fn()
            except Exception:
                log.exception("error closing %s", name)
        for th in self.ctl.leave_threads:
            th.join(timeout=3.0)
        release_single_instance(self._mutex)
        self._mutex = None

    # ------------------------------------------------------------ one frame
    def step(self, pkt: FramePacket, sig: Signals) -> None:
        t_loop = time.perf_counter()
        now = self.clock.now()
        self._frames += 1
        self._last_sig = sig
        sfx = self.sfx or NullSfx()
        if sig.face is not None and sig.face.fresh:
            self._last_face, self._last_face_t = sig.face, now
        last_face = self._last_face if now - self._last_face_t < LAST_FACE_S else None
        ctx = FxContext(pkt.frame, sig, now, self.assets, sfx, self.plate, last_face=last_face)
        if not self._first_frame_done:
            self._first_frame(pkt)
        if self.hotkeys is not None:
            self._commands.extend(self.hotkeys.drain())
        while self._commands:
            self.command(self._commands.popleft(), ctx)
        self._feed_plate(pkt, sig, now)
        if sig.face is not None and sig.face.fresh:
            faceswap.service().offer(pkt.frame, sig.face.lm, now)  # learns your face for face-swap memes

        t0 = time.perf_counter()
        level = self.supervisor.level
        busy_until = getattr(self.sfx, "busy_until", 0.0)
        events = self.snaps.step(sig, now, busy_until) if self.snaps is not None and level < 2 else []
        truths = self.det.update(sig, sig.t)
        if self._auto_calib and not self.calibrating:
            self._first_use(sig, now, ctx)
        self._truths = truths
        if self.calibrating:
            self._calibrate_tick(sig, now, truths)
        elif not self.arbiter.armed and any(on and self.ctl.target_for(g) != "none" for g, on in truths.items()):
            # a gesture while they're switched off: say so, big (nothing would happen otherwise)
            self._show_banner("Gestures are OFF", "press A in this window (or Ctrl+Alt+A) to turn them on", 0.0,
                              now, hold_s=3.0, state="wait")
        if self._quit_at is not None and now >= self._quit_at:
            self._quit = True
        arb = self.arbiter
        arb.presence_state = self.presence.state
        arb.modal_active = self.ctl.modal_active
        arb.card_active = self.ctl.card_active
        intents = arb.update(truths, events, sig.t, busy=self.compositor.busy_with_look(),
                             snap_loaded=self.det.snap_loaded, t_on=self.det.t_on) if level < 2 else []
        if self.calibrating:  # calibrating: the gestures you're asked to show must not start effects
            intents = []
        for it in intents:
            self.ctl.handle(it, ctx)
        self.ctl.follow_charge(ctx)
        if self._sig_dump is not None:
            self._dump_signals(sig, now, truths, intents)
        if self.opts.try_target:
            self._try_tick(ctx)
        t1 = time.perf_counter()

        out = pkt.frame if level >= 2 else self.compositor.render(ctx)
        t2 = time.perf_counter()
        self._send(out)
        t3 = time.perf_counter()
        if self.worker is not None:
            self.worker.set_needs(*self._needs())
        if self.preview is not None and self.preview.due():
            self._handle_key(self.preview.show(out, self._hud(now)))
        if self._dump_dir and now >= self._dump_next:
            self._dump_next = now + DUMP_EVERY_S
            self._dump_preview(out, now)
        t4 = time.perf_counter()

        render_ms = (t2 - t1) * 1e3
        for key, ms in (("gestures", t1 - t0), ("render", t2 - t1), ("send", t3 - t2), ("preview", t4 - t3),
                        ("loop", t4 - t_loop)):
            self.stats.add(key, ms * 1e3)
        self.supervisor.frame(now, render_ms)
        # the calibration analysis briefly competes for the CPU: not a reason to shut effects down
        changed = None if self.calibrating else self.supervisor.check(
            now, self.cam.fps if self.cam is not None else 30.0, self.compositor.error_times)
        if changed is not None:
            self._apply_level(changed, ctx)
        if now - self._last_stats >= STATS_EVERY_S:
            self._log_stats(now)

    def _send(self, frame: np.ndarray) -> None:
        if self.vcam is not None:
            self.vcam.send(frame)

    def _needs(self) -> tuple[bool, bool, bool]:
        """Which models the next frames must run regardless of their cadence."""
        face = self.compositor.needs_face() or self.calibrating or any(
            self.arbiter.dwells[g].state == ARMING for g in FACE_POSES if g in self.arbiter.dwells
        )
        mask = self.compositor.needs_mask() or self.plate.capturing or self._plate_at is not None \
            or self.compositor.find("charge") is not None  # a vanish may be seconds away: keep the mask fresh
        hands = self.snaps is not None and self.snaps.pending
        return face, mask, hands

    def _idle(self) -> None:
        """No camera frame for a while: tell the meeting, keep hotkeys and the window alive."""
        now = self.clock.now()
        if self._deadline is not None and now >= self._deadline:
            self._quit = True
            return
        if now - self._last_card < STALL_CARD_EVERY_S:
            return
        self._last_card = now
        from zap.ui.preview import info_card

        err = self.cam.error if self.cam is not None else None
        card = info_card(self.W, self.H, "Camera reconnecting...", [err or "waiting for the webcam"])
        self._send(card)
        if self.hotkeys is not None:
            self._commands.extend(self.hotkeys.drain())  # applied with the next frame
        if self.preview is not None:
            self._handle_key(self.preview.show(card, self._hud(now)))

    def _handle_key(self, action: str | None) -> None:
        if action == "quit":
            self._quit = True
        elif action == "panic":
            self._commands.append("panic")
        elif action in ("cancel", "skip", "marks", "arm_toggle"):
            self._commands.append(action)

    # ------------------------------------------------------------ commands
    def command(self, cmd: str, ctx: FxContext) -> None:
        """A hotkey (or preview key) command, applied on the frame loop."""
        now = ctx.now
        if cmd == "cancel":  # Esc in the preview: stop calibrating, else stop leaving, else end the card
            if self._calib is not None or self._auto_calib:
                self._calib, self._auto_calib, self._banner = None, False, None
                self.notices.add("calibration cancelled (Ctrl+Alt+C to do it later)", "info", 6, key="calib")
                if self.opts.quit_after_calibration:
                    self._quit_at = now + 1.0
            elif not self.ctl.cancel_leave(ctx):
                self.ctl.finish_card(ctx)
            return
        for it in self.arbiter.command(cmd, now):
            self.ctl.handle(it, ctx)
        if cmd == "arm_toggle":
            self.notices.add("gestures ARMED" if self.arbiter.armed else "gestures disarmed", key="arm")
            self._show_banner("Gestures ON" if self.arbiter.armed else "Gestures OFF",
                              "live in 3 s" if self.arbiter.armed else "press A to turn them back on", 1.0, now,
                              hold_s=2.0, state="tick" if self.arbiter.armed else "wait")
        elif cmd == "sound_toggle" and self.sfx is not None:
            self.sfx.set_enabled(not self.sfx.enabled)
            self.notices.add(f"sound effects {'on' if self.sfx.enabled else 'off'}", key="sound")
        elif cmd == "plate":
            self._plate_at = now + PLATE_COUNTDOWN_S
            self.notices.add(f"Step out of view! Capturing the empty background in {PLATE_COUNTDOWN_S:.0f} s",
                             "warn", PLATE_COUNTDOWN_S + PLATE_CAPTURE_S + 1, key="plate")
        elif cmd == "reload":
            self.reload()
        elif cmd == "calibrate" and not self.calibrating:  # a repeated key press doesn't restart it
            self._calib = Calibration(now)
            self._calib_shown = ""
            log.info("calibration started")
        elif cmd == "skip" and self._calib is not None:
            self._calib.skip(now)
        elif cmd == "marks":
            self._show_marks = not self._show_marks
        elif cmd in ("sens_up", "sens_down") and self.snaps is not None:
            d = self.snaps.det
            d.sensitivity = d.sensitivity + (1 if cmd == "sens_up" else -1)
            self.snaps.fusion.sensitivity = d.sensitivity
            self.notices.add(f"snap sensitivity {d.sensitivity} (1 strict .. 5 loose)", key="sens")

    @property
    def calibrating(self) -> bool:
        return self._calib is not None or self._calib_job is not None

    def _first_use(self, sig: Signals, now: float, ctx: FxContext) -> None:
        """No calibration saved yet (or "zap calibrate"): start one once a face is steadily in view."""
        if self.preview is None:  # the steps are shown in the preview; without it, say how to do it later
            self._auto_calib = False
            log.info("not calibrated yet: run 'zap calibrate' (or press Ctrl+Alt+C with the preview open)")
            return
        if sig.face is None:
            self._face_since = None
            self._show_banner("Welcome to ZAP!", "Sit where you usually do and look at the camera", 0.0, now)
            return
        self._face_since = self._face_since if self._face_since is not None else now
        wait = FIRST_USE_FACE_S - (now - self._face_since)
        self._show_banner("Welcome to ZAP! First, a 1-minute calibration to YOUR face",
                          f"starting in {max(0.0, wait):.0f} s - just follow the prompts here", 0.0, now)
        if wait <= 0:
            self._auto_calib = False
            self.command("calibrate", ctx)

    def _show_banner(self, title: str, sub: str, progress: float, now: float, hold_s: float = 0.6,
                     state: str = "") -> None:
        self._banner, self._banner_until = (title, sub, progress, state), now + hold_s

    def _calibrate_tick(self, sig: Signals, now: float, truths: dict[str, bool]) -> None:
        """Ctrl+Alt+C (see :mod:`zap.gestures.calibrate`): each step waits until the camera sees you do it;
        then the analysis runs on a worker; gestures that still fail get one more try."""
        run = self._calib
        if run is not None:
            view = run.feed(sig, self.det, truths, now)
            if view is not None:
                if run.step is not None and run.step.name != self._calib_shown and view.state != "lead":
                    self._calib_shown = run.step.name
                    log.info("calibration step %d/%d: %s", view.n, view.total, run.step.name)
                sub = view.sub if view.state == "lead" else f"step {view.n} of {view.total}:  {view.sub}"
                self._show_banner(view.title, sub, view.progress, now, state=view.state)
                return
            self._calib = None
            job: dict[str, Any] = {"result": None, "error": None, "run": run}
            self._calib_job = job
            _save_session(run.frames)
            threading.Thread(target=_calibration_worker, args=(job, list(run.frames), dict(self.cfg.gestures),
                                                               set(run.skipped)),
                             name="zap-calibrate", daemon=True).start()
            return
        job = self._calib_job
        if job is None:
            return
        if job["result"] is None and job["error"] is None:
            self._show_banner("Checking what the camera saw ...", "a few seconds", 1.0, now, state="seen")
            return
        self._calib_job = None
        if job["error"] is not None:
            self.notices.add(f"calibration failed: {job['error']}", "error", 12, key="calib")
            if self.opts.quit_after_calibration:
                self._quit_at = now + CALIB_RESULT_S
            return
        result, run = job["result"], job["run"]
        if result.failed and run.retry(result.failed, now):  # recheck what didn't work, once
            names = ", ".join(calibrate.NAMES[g] for g in result.failed)
            log.info("calibration: retrying %s", names)
            self.notices.add(f"Not recognised yet: {names}. Let's try those once more.", "warn", 8, key="calib")
            self._calib = run
            return
        if self.opts.quit_after_calibration:
            self._quit_at = now + CALIB_RESULT_S
        cal = result.calibration()
        if cal["brows"] is None:  # couldn't calibrate the eyebrows this time: keep the last good one
            cal["brows"] = (_load_calibration() or {}).get("brows")
        self.det.apply_calibration(cal)
        _save_calibration(cal)
        good = [calibrate.NAMES[g] for g, v in result.verdicts.items() if v.ok]
        bad = [v.line() for v in result.verdicts.values() if not v.ok]
        self.notices.clear("calib")
        self.notices.add("CALIBRATED. Working: " + (", ".join(good) or "none"), "info", 25, key="calib")
        for i, line in enumerate(bad[:5]):
            self.notices.add(line + " (Ctrl+Alt+C to redo)", "error", 25, key=f"calib{i}")
        self._show_banner("Calibrated to you!" if not bad else "Calibrated - see below what still needs work",
                          f"{len(good)} of {len(result.verdicts)} gestures recognised. Try them now: "
                          "each row on the left lights up when ZAP sees it", 1.0, now, hold_s=CALIB_RESULT_S,
                          state="tick" if not bad else "wait")

    def reload(self) -> None:
        try:
            new = zap_config.load(self.cfg.source)
            det = PoseDetector(new.gestures)
            det.apply_calibration(_load_calibration())
        except (ConfigError, OSError) as e:
            self.notices.add(f"config error, kept the old settings: {e}", "error", 20, key="config")
            return
        new.app.armed_on_start = self.cfg.app.armed_on_start
        self.cfg = new
        self.det = det
        self.arbiter.configure(new)
        self.ctl.cfg = new
        self.assets = AssetResolver()
        if self.sfx is not None:
            self.sfx.assets = self.assets
            self.sfx.set_volume(new.sound.volume)
            if self.sfx.enabled != new.sound.enabled:
                self.sfx.set_enabled(new.sound.enabled)
        if self.snaps is not None:
            self.snaps.det.sensitivity = new.snap.sensitivity
            self.snaps.fusion.sensitivity = self.snaps.det.sensitivity
        if self.perception is not None:
            self.perception.face_every = max(1, new.perception.face_every)
            self.perception.mask_every = max(1, new.perception.mask_every)
            self.perception.hands_idle_every = max(1, new.perception.hands_idle_every)
        self.notices.clear("config")
        self.notices.add("config and meme files reloaded", key="reload")

    # ------------------------------------------------------------ background plate
    def _plate_path(self) -> Path:
        return paths.calibration_dir() / PLATE_FILE

    def _first_frame(self, pkt: FramePacket) -> None:
        self._first_frame_done = True
        # the camera can take ~10 s to open (DirectShow, then Media Foundation): judge fps from now
        self.supervisor = Supervisor(self.clock.now())
        if self.cam is not None and self.cam.info is not None:
            log.info("capturing from %s", self.cam.info)
        if self.plate.load(self._plate_path(), pkt.frame):
            self._plate_clean = True
            self.notices.add("loaded your saved empty-room background", key="plate")

    def _feed_plate(self, pkt: FramePacket, sig: Signals, now: float) -> None:
        plate = self.plate
        if self._plate_at is not None and now >= self._plate_at:
            self._plate_at = None
            self._plate_person = 0.0
            plate.capture_clean(PLATE_CAPTURE_S, now)
            self.notices.add("capturing the background... stay out of view", "warn", PLATE_CAPTURE_S + 0.5,
                             key="plate")
        if plate.capturing or self._plate_was_capturing:
            plate.update(pkt.frame, sig.mask, now)
            if sig.mask is not None and sig.mask_fresh:
                self._plate_person = max(self._plate_person, float(np.mean(sig.mask > 0.5)))
            if self._plate_was_capturing and not plate.capturing:
                self._finish_capture()
            self._plate_was_capturing = plate.capturing
            return
        if sig.mask_fresh and sig.mask is not None and now - self._plate_fed >= PLATE_EVERY_S:
            self._plate_fed = now
            plate.update(pkt.frame, sig.mask, now)

    def _finish_capture(self) -> None:
        if self._plate_person > PLATE_MAX_PERSON:
            self.plate.reset()
            self.notices.add("You were still in view, so that background was not kept. Try Ctrl+Alt+B again.",
                             "error", 12, key="plate")
            return
        self._plate_clean = True
        saved = self.plate.save(self._plate_path())
        self.notices.add("clean background captured" + (" and saved for next time" if saved else ""), key="plate")

    # ------------------------------------------------------------ zap try
    def _try_tick(self, ctx: FxContext) -> None:
        target = self.opts.try_target
        now = ctx.now
        if self._try_release_at is not None and now >= self._try_release_at:
            self._try_release_at = None
            if target == HANDS_UP:
                self.ctl.release(HANDS_UP, ctx)
            else:
                self.compositor.release(target, ctx)
        if now < self._next_try:
            return
        self._next_try = now + self.opts.try_every_s
        if target in ALL_GESTURES:
            self.ctl.handle(Intent("trigger", target, now, _palm(ctx)), ctx)
        else:
            self.ctl.start_target(target, ctx, _palm(ctx))
        if target in (HANDS_UP, "cinema"):
            self._try_release_at = now + self.opts.try_hold_s

    # ------------------------------------------------------------ supervisor
    def _apply_level(self, level: int, ctx: FxContext) -> None:
        p = self.perception
        base_face = max(1, self.cfg.perception.face_every)
        if level >= 1:
            if p is not None:
                p.face_every = max(base_face, 3)
            self.presence.set_stride(3)
        else:
            if p is not None:
                p.face_every = base_face
            self.presence.set_stride(2)
        if self.worker is not None:
            self.worker.passthrough = level >= 2
        if level >= 2:
            for it in self.arbiter.command("disarm", ctx.now):
                self.ctl.handle(it, ctx)
            self.ctl.panic(ctx)
            self.notices.add(f"too slow ({self.supervisor.reason}): plain camera, gestures off. "
                             "Close other apps; ZAP retries on its own.", "error", 30, key="degrade")
        else:
            self.notices.add(f"performance level L{level} ({self.supervisor.reason})", "warn" if level else "info",
                             10, key="degrade")
        log.warning("degrade level -> L%d (%s)", level, self.supervisor.reason)

    # ------------------------------------------------------------ HUD and stats
    def _dump_signals(self, sig: Signals, now: float, truths: dict[str, bool], intents: list) -> None:
        from zap.recording import frame_record

        rec = frame_record(sig, now - self._run_t0)
        rec["truths"] = sorted(g for g, on in truths.items() if on)
        rec["suppressed"] = dict(self.arbiter.suppressed)
        rec["intents"] = [f"{i.kind}:{i.gesture}" for i in intents]
        rec["why"] = dict(self.det.debug.get("sign_why", {}))
        rec["brow"] = self.det.debug.get(EYEBROWS_UP)
        try:
            self._sig_dump.write(json.dumps(rec, default=str) + "\n")
        except (OSError, ValueError):
            self._sig_dump = None

    def _dump_preview(self, out: np.ndarray, now: float) -> None:
        """Dev aid (ZAP_PREVIEW_DUMP=<folder>): save what the preview shows, a few times a minute."""
        import cv2

        from zap.ui.preview import compose

        try:
            folder = Path(self._dump_dir)
            folder.mkdir(parents=True, exist_ok=True)
            view = compose(out, self._hud(now), mirror=self.cfg.app.preview_mirror and not self.cfg.camera.mirror)
            name = f"preview-{int(now - self._run_t0):05d}.jpg"
            cv2.imwrite(str(folder / name), view, [cv2.IMWRITE_JPEG_QUALITY, 80])
        except (OSError, cv2.error):
            log.exception("could not save the preview")
            self._dump_dir = None

    def _hud(self, now: float) -> Any:
        from zap.ui.preview import HudState, marks_from, pose_rows

        st = self.arbiter.status(now)
        hud = HudState(
            armed=st["armed"], grace_left=st["grace_left"], leave_guard_left=st["leave_guard_left"],
            presence=self.presence.state, level=self.supervisor.level,
            fps_cam=self.cam.fps if self.cam is not None else 0.0, fps_out=self.supervisor.fps(now),
            dwell={g: (d["state"], d["progress"], d["cooldown_left"]) for g, d in st["dwell"].items()},
            suppressed={**self.det.debug.get("sign_why", {}), **st["suppressed"]},
            effects=[e.name for e in self.compositor.active() if e.layer != Layer.PRESENCE],
        )
        if self.cam is not None and self.cam.info is not None:
            hud.camera = f"{self.cam.info.name} ({self.cam.info.backend_name})"
        try:
            if self.perception is not None:
                for k in ("hands", "face", "mask"):
                    hud.stages[k] = self.perception.p95(k)
                hud.stages["perception"] = self.perception.p95("total")
            for k in ("gestures", "render", "send", "loop"):
                hud.stages[k] = self.stats.p95(k)
        except RuntimeError:  # a stats window changed under us; next refresh will do
            pass
        sn = self.snaps
        if sn is not None:
            hud.snap_mode = ("camera (LOADED)" if sn.fusion.loaded_now else "camera") if sn.vision_only                 else "mic + camera"
            if sn.mic is not None and not sn.mic.error:
                hud.mic_level, hud.mic_floor = sn.mic.level, sn.mic.noise_floor
            hud.snaps = list(sn.recent) + [(False, r) for r in list(sn.fusion.rejects)[-2:]]
            hud.burst = sn.bursts.count
        hud.rows = pose_rows(self.cfg.bindings)
        hud.live = {g for g, on in self._truths.items() if on}
        if self._banner is not None and now < self._banner_until:
            hud.banner = self._banner
        if (self.calibrating or self._show_marks) and self._last_sig is not None:
            hud.marks = marks_from(self._last_sig)
        b = self.det.debug.get(EYEBROWS_UP)
        if b and "level" in b and self.ctl.target_for(EYEBROWS_UP) != "none":
            hud.brow = (b["level"], b["base"], b["threshold"])
        lv = self.ctl.leaving
        if lv is not None:
            hud.leave = f"LEAVING in {lv.seconds_left:.1f}s (snap / Esc cancels)" if lv.state == "counting" \
                else f"leave: {lv.state}"
        if self.sfx is not None:
            hud.sound = f"sound {'on' if self.sfx.enabled else 'OFF'} ({self.sfx.mode})"
        if self.plate.capturing:
            hud.plate = f"capturing background {self.plate.progress * 100:.0f}%"
        else:
            hud.plate = f"background {'clean' if self._plate_clean else f'{self.plate.coverage * 100:.0f}% seen'}"
        hud.warnings = self._warnings()
        return hud

    def _warnings(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if self.cam is not None and self.cam.error:
            out.append(("error", f"camera: {self.cam.error}"))
        f = self._last_face
        if f is not None and f.lm[FACE_CHIN_IDX, 1] > LOW_CHIN * self.H:  # a real user's chin sat at 83%
            out.append(("warn", ("Your collar is at the bottom edge of the picture, so hands there can't be seen "
                                 "(Harvey). Tilt the screen/camera up a little or sit back.")))
        sn = self.snaps
        if sn is not None and not sn.chose_vision and (sn.mic is None or sn.mic.error):
            why = sn.mic.error if sn.mic is not None else "not available"
            out.append(("warn", f"MIC OFF: {why}. Snaps are seen by the camera only."))
        if self.sfx is not None and self.sfx.warning:
            out.append(("warn", self.sfx.warning))
        if faceswap.service().status:
            out.append(("info", faceswap.service().status))
        if enhance.enhancer().status:
            out.append(("info", enhance.enhancer().status))
        if self.assets.missing:
            names = ", ".join(sorted(Path(m).name for m in self.assets.missing))
            out.append(("info", f"stand-ins used; add your own memes to {paths.memes_dir()}: {names}"))
        if self.presence.state != "visible" and not self._plate_clean:
            out.append(("info", "Tip: press Ctrl+Alt+B and step out of view for 3 s to capture a clean background"))
        return out + self.notices.active()

    def _log_stats(self, now: float) -> None:
        self._last_stats = now
        fps = self.supervisor.fps(now)
        self._fps_hist.append(fps)
        p = self.perception
        try:
            perc = " ".join(f"{k} {p.p95(k):.1f}" for k in ("hands", "face", "mask", "total")) if p else "-"
            main = " ".join(f"{k} {self.stats.p95(k):.1f}" for k in MAIN_STAGES)
        except RuntimeError:
            return
        w = self.worker
        log.info("stats: cam %.1f fps, out %.1f fps | perception p95 ms: %s | main p95 ms: %s | L%d, %s, %s, "
                 "dropped %d", self.cam.fps if self.cam else 0.0, fps, perc, main, self.supervisor.level,
                 self.presence.state, "armed" if self.arbiter.armed else "disarmed", w.dropped if w else 0)

    def _log_summary(self) -> None:
        if not self._frames:
            return
        secs = max(1e-6, self.clock.now() - self._run_t0)
        log.info("run summary: %d frames in %.1f s = %.1f fps output; perception p95 total %.1f ms; "
                 "render p95 %.1f ms; send p95 %.1f ms; loop p95 %.1f ms", self._frames, secs, self._frames / secs,
                 self.perception.p95("total") if self.perception else 0.0, self.stats.p95("render"),
                 self.stats.p95("send"), self.stats.p95("loop"))


def _palm(ctx: FxContext) -> tuple[float, float] | None:
    if ctx.sig.hands:
        pc = ctx.sig.hands[0].palm_center
        return float(pc[0]), float(pc[1])
    return None


# ---------------------------------------------------------------- zap camtest
def run_camtest(cfg: Config, seconds: float | None = None, preview: bool = True) -> int:
    """Camera -> virtual camera with no effects, showing the fps: checks the video path only."""
    from zap.devices.camera import Camera
    from zap.devices.vcam import VirtualCam
    from zap.ui.preview import HudState, Preview

    mutex = acquire_single_instance()
    vcam = VirtualCam(cfg.output.width, cfg.output.height, cfg.output.fps)
    cam = Camera.from_config(cfg).start()
    win = Preview(mirror=cfg.app.preview_mirror and not cfg.camera.mirror) if preview else None
    sup = Supervisor(time.perf_counter())
    send = Timings(("send",), window=150)
    t0 = last = time.perf_counter()
    n = 0
    try:
        while seconds is None or time.perf_counter() - t0 < seconds:
            pkt = cam.wait_latest(1.0)
            now = time.perf_counter()
            if pkt is None:
                log.info("waiting for the camera: %s", cam.error or "no frames yet")
                continue
            ts = time.perf_counter()
            vcam.send(pkt.frame)
            send.add("send", (time.perf_counter() - ts) * 1e3)
            sup.frame(now, 0.0)
            n += 1
            if win is not None and win.due():
                info = f"{cam.info.name} ({cam.info.backend_name})" if cam.info else ""
                hud = HudState(fps_cam=cam.fps, fps_out=sup.fps(now), camera=info,
                               stages={"send": send.p95("send")}, presence="camtest",
                               hotkeys="camtest: camera -> OBS Virtual Camera, no effects  |  q quit")
                if win.show(pkt.frame, hud) == "quit":
                    break
            if now - last >= STATS_EVERY_S:
                last = now
                log.info("camtest: cam %.1f fps, out %.1f fps, send p95 %.1f ms (%s)", cam.fps, sup.fps(now),
                         send.p95("send"), cam.info)
    except KeyboardInterrupt:
        pass
    finally:
        secs = time.perf_counter() - t0
        log.info("camtest: %d frames in %.1f s (%.1f fps)", n, secs, n / max(secs, 1e-6))
        cam.stop()
        vcam.close()
        if win is not None:
            win.close()
        release_single_instance(mutex)
    return EXIT_OK


# ---------------------------------------------------------------- calibration (Ctrl+Alt+C)
CALIB_FILE = "gestures.json"
LEGACY_BROW_FILE = "eyebrows.json"  # the eyebrow-only calibration of earlier versions
CALIB_LEAD_S = 3.0  # time to read the intro before the first step
FIRST_USE_FACE_S = 3.0  # first use: a face this long in view starts the calibration
CALIB_RESULT_S = 15.0  # the result stays up this long ("zap calibrate" then exits)


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _load_calibration() -> dict | None:
    """The saved Ctrl+Alt+C result, or an older eyebrow-only one as {"brows": ...}, or None."""
    cal = _read_json(paths.calibration_dir() / CALIB_FILE)
    if cal is not None and cal.get("version") == calibrate.VERSION:
        return cal
    brows = _read_json(paths.calibration_dir() / LEGACY_BROW_FILE)
    return {"brows": brows} if brows is not None and "rise" in brows else None


def _save_calibration(cal: dict) -> None:
    f = paths.calibration_dir() / CALIB_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(cal, indent=1), encoding="utf-8")


def _save_session(frames: list) -> None:
    """Dev aid: ZAP_CALIB_SAVE=<folder> keeps the calibration steps' landmarks (no video) for offline tuning."""
    folder = os.environ.get("ZAP_CALIB_SAVE")
    if not folder or not frames:
        return
    from zap.recording import frame_record

    try:
        out = Path(folder)
        out.mkdir(parents=True, exist_ok=True)
        t0 = frames[0].sig.t
        path = out / time.strftime("calibration-%Y%m%d-%H%M%S.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps({**frame_record(f.sig, f.sig.t - t0, f.step), "use": f.use}) + "\n"
                          for f in frames)
        log.info("calibration landmarks saved to %s", path)
    except OSError:
        log.exception("could not save the calibration landmarks")


def _calibration_worker(job: dict[str, Any], frames: list, gestures: dict, skipped: set[str]) -> None:
    try:
        job["result"] = calibrate.analyse(frames, gestures, skipped=skipped)
    except Exception as e:  # reported on the HUD; the old calibration stays in use
        log.exception("calibration analysis failed")
        job["error"] = str(e) or type(e).__name__
