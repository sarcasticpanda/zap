"""The local preview window with the HUD. Only you see it; nothing here reaches Zoom.

The HUD is drawn on a downscaled COPY of the outgoing frame, never on the frame
that is sent, and the window is refreshed at most ``max_fps`` times a second so
it costs the frame loop a couple of milliseconds at most. OpenCV's window must
be pumped from the thread that created it, so ``show`` is called from the
frame loop; it also returns the key the user pressed in the window.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)

TITLE = "ZAP preview (only you see this)"
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GRAY, DIM = (240, 240, 240), (170, 170, 170), (110, 110, 110)
GREEN, YELLOW, RED, ORANGE, CYAN = (90, 220, 90), (60, 210, 240), (70, 70, 235), (40, 150, 255), (230, 200, 80)
KEY_ACTIONS = {27: "cancel", 32: "panic", ord("q"): "quit", ord("Q"): "quit"}
# HUD rows: which sign does what (the preview font has no emoji, so the signs are spelled out)
SIGN_NAMES = {"two_thumbs_up": "thumbs up", "peace": "peace", "point_up": "1 finger up", "thumbs_down": "thumbs down",
              "rock": "rock on", "hands_up": "hands up", "collar": "collar", "cover_face": "cover face",
              "forehead": "forehead", "eyebrows_up": "brows up"}
EFFECT_NAMES = {"money": "money", "outro": "to be cont.", "brb": "right back", "cinema": "cinema", "sad": "sad MJ",
                "presence": "vanish", "cutaway:michael_scott": "michael"}


def pose_rows(bindings: dict[str, str]) -> list[tuple[str, str]]:
    """(gesture, "sign > effect") for every held gesture that is bound to something."""
    rows = []
    for g, sign in SIGN_NAMES.items():
        target = bindings.get(g, "none") or "none"
        if target != "none":
            rows.append((g, f"{sign} > {EFFECT_NAMES.get(target, target.split(':')[-1].replace('_', ' '))}"))
    return rows


@dataclass
class HudState:
    """Everything the HUD shows, collected by the app once per preview refresh."""

    armed: bool = False
    grace_left: float = 0.0
    leave_guard_left: float = 0.0
    presence: str = "visible"
    level: int = 0  # degrade level 0..2
    camera: str = ""  # name (backend)
    fps_cam: float = 0.0
    fps_out: float = 0.0
    stages: dict[str, float] = field(default_factory=dict)  # name -> ms (p95)
    mic_level: float | None = None  # dBFS; None = microphone off
    mic_floor: float = -100.0
    snap_mode: str = ""  # "audio+video" / "vision only"
    snaps: list[tuple[bool, str]] = field(default_factory=list)  # last candidates: (accepted, reason)
    burst: int = 0  # snaps counted in the open burst
    dwell: dict[str, tuple[str, float, float]] = field(default_factory=dict)  # g -> (state, progress, cooldown)
    suppressed: dict[str, str] = field(default_factory=dict)
    rows: list[tuple[str, str]] = field(default_factory=list)  # (gesture, "sign > effect"), see pose_rows
    brow: tuple[float, float, float] | None = None  # level, baseline, threshold
    effects: list[str] = field(default_factory=list)
    outro: str = ""
    sound: str = ""
    plate: str = ""
    warnings: list[tuple[str, str]] = field(default_factory=list)  # (severity, text): error / warn / info
    hotkeys: str = "Ctrl+Alt+A arm  P panic  1-8 effects  B background  S sound  R reload  |  window: Esc cancel, Space panic, q quit"


class Preview:
    """cv2 window at most ``max_fps`` times a second. ``show`` returns "cancel", "panic", "quit" or None."""

    def __init__(self, mirror: bool = True, max_fps: float = 15.0, size: tuple[int, int] = (960, 540),
                 title: str = TITLE) -> None:
        self.mirror = mirror
        self.min_dt = 1.0 / max(1.0, max_fps)
        self.size = size
        self.title = title
        self._last = 0.0
        self._small = np.empty((size[1], size[0], 3), np.uint8)
        self._view = np.empty_like(self._small)
        self._open = False
        self.closed = False  # the user closed the window

    def due(self, now: float | None = None) -> bool:
        return (time.perf_counter() if now is None else now) - self._last >= self.min_dt

    def show(self, frame: np.ndarray, hud: HudState | None) -> str | None:
        """Draw if a refresh is due; always pumps window events. Returns the key action, if any."""
        now = time.perf_counter()
        if not self._open:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.title, *self.size)
            self._open = True
        if now - self._last >= self.min_dt:
            self._last = now
            cv2.resize(frame, self.size, dst=self._small, interpolation=cv2.INTER_AREA)
            if self.mirror:
                cv2.flip(self._small, 1, dst=self._view)
            else:
                np.copyto(self._view, self._small)
            if hud is not None:
                draw_hud(self._view, hud)
            cv2.imshow(self.title, self._view)
        key = cv2.waitKey(1)
        if self._window_gone():
            self.closed = True
            return "quit"
        return KEY_ACTIONS.get(key & 0xFF) if key >= 0 else None

    def _window_gone(self) -> bool:
        try:
            return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def close(self) -> None:
        if self._open:
            try:
                cv2.destroyWindow(self.title)
                cv2.waitKey(1)
            except cv2.error:
                pass
            self._open = False


# ---------------------------------------------------------------- drawing
class PreviewThread:
    """A :class:`Preview` drawn on its own thread, so the window (``cv2.waitKey`` alone can block for a
    15 ms Windows timer tick) never slows the video the meeting sees. Same interface as Preview;
    keys come back on the next ``show`` call."""

    def __init__(self, **kw) -> None:
        self._inner = Preview(**kw)
        self._slot: tuple[np.ndarray, HudState | None] | None = None
        self._keys: deque[str] = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="zap-preview", daemon=True)
        self._thread.start()

    @property
    def closed(self) -> bool:
        return self._inner.closed

    def due(self, now: float | None = None) -> bool:
        return self._inner.due(now)

    def show(self, frame: np.ndarray, hud: HudState | None) -> str | None:
        small = cv2.resize(frame, self._inner.size, interpolation=cv2.INTER_AREA)  # also a private copy
        with self._lock:
            self._slot = (small, hud)
            return self._keys.popleft() if self._keys else None

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                with self._lock:
                    item, self._slot = self._slot, None
                if item is None:
                    if self._inner._open:
                        key = cv2.waitKey(5)
                        action = KEY_ACTIONS.get(key & 0xFF) if key >= 0 else None
                        if self._inner._window_gone():
                            self._inner.closed, action = True, "quit"
                        if action:
                            with self._lock:
                                self._keys.append(action)
                    else:
                        time.sleep(0.005)
                    continue
                self._inner._last = 0.0  # due() was checked by the caller
                action = self._inner.show(*item)
                if action:
                    with self._lock:
                        self._keys.append(action)
        finally:
            self._inner.close()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def _text(img: np.ndarray, s: str, x: int, y: int, color=WHITE, scale: float = 0.45, thick: int = 1) -> int:
    """Draw with a dark outline for contrast on any background; returns the text width."""
    cv2.putText(img, s, (x, y), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
    return cv2.getTextSize(s, FONT, scale, thick)[0][0]


def _shade(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, alpha: float = 0.55) -> None:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        cv2.convertScaleAbs(roi, dst=roi, alpha=1.0 - alpha)


def _bar(img: np.ndarray, x: int, y: int, w: int, h: int, frac: float, color, marks=()) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    f = min(1.0, max(0.0, frac))
    if f > 0:
        cv2.rectangle(img, (x, y), (x + int(w * f), y + h), color, -1)
    for m, mc in marks:
        mx = x + int(w * min(1.0, max(0.0, m)))
        cv2.line(img, (mx, y - 2), (mx, y + h + 2), mc, 2)


def _check(img: np.ndarray, x: int, y: int, color=GREEN) -> None:
    """A small tick mark (Hershey fonts have no check-mark glyph)."""
    cv2.line(img, (x, y - 4), (x + 3, y), color, 2, cv2.LINE_AA)
    cv2.line(img, (x + 3, y), (x + 10, y - 9), color, 2, cv2.LINE_AA)


def draw_hud(img: np.ndarray, s: HudState) -> None:
    H, W = img.shape[:2]
    _shade(img, 0, 0, 360, 190 + 15 * len(s.rows) + (16 if s.brow is not None else 0))

    # status badge
    if not s.armed:
        badge, color = "DISARMED", GRAY
    elif s.grace_left > 0:
        badge, color = f"ARMING {s.grace_left:.1f}s", YELLOW
    else:
        badge, color = "ARMED", GREEN
    x = 10 + _text(img, badge, 10, 24, color, 0.7, 2) + 12
    pres_color = WHITE if s.presence == "visible" else CYAN
    x += _text(img, s.presence.upper(), x, 24, pres_color, 0.55, 1) + 12
    if s.level:
        _text(img, f"DEGRADED L{s.level}", x, 24, RED if s.level >= 2 else ORANGE, 0.55, 1)

    y = 46
    _text(img, f"cam {s.fps_cam:4.1f} fps  out {s.fps_out:4.1f} fps", 10, y)
    if s.camera:
        _text(img, s.camera[:44], 10, y + 16, DIM, 0.38)
    y += 34
    st = s.stages
    perc = "  ".join(f"{k} {st[k]:.0f}" for k in ("hands", "face", "mask", "perception") if k in st)
    main = "  ".join(f"{k} {st[k]:.1f}" for k in ("gestures", "render", "send", "loop") if k in st)
    _text(img, f"p95 ms {perc}", 10, y, GRAY, 0.4)
    _text(img, f"       {main}", 10, y + 15, GRAY, 0.4)
    y += 34

    # microphone
    if s.mic_level is None:
        _text(img, "MIC OFF", 10, y, RED, 0.45, 1)
        _text(img, f"snaps: {s.snap_mode or 'camera'}", 80, y, GRAY, 0.4)
    else:
        lo = -80.0
        frac = (s.mic_level - lo) / -lo
        _text(img, "mic", 10, y, GRAY, 0.4)
        _bar(img, 42, y - 9, 150, 9, frac, GREEN, [((s.mic_floor - lo) / -lo, YELLOW)])
        _text(img, f"{s.mic_level:4.0f} dB  floor {s.mic_floor:4.0f}", 198, y, GRAY, 0.38)
    y += 18
    _text(img, "snaps", 10, y, GRAY, 0.4)
    x = 58
    for ok, reason in s.snaps[-5:]:
        if ok:
            _check(img, x, y)
            x += 16
        else:
            x += _text(img, reason, x, y, DIM, 0.36) + 8
    if s.burst:
        _text(img, f"x{s.burst}", x + 4, y, ORANGE, 0.45, 1)
    y += 20

    # pose dwell bars: "sign > effect", how long it has been held, and why it can't fire
    for g, label in s.rows:
        state, prog, cool = s.dwell.get(g, ("idle", 0.0, 0.0))
        color = GREEN if state == "active" else YELLOW if state == "arming" else DIM
        _text(img, label[:26], 10, y, WHITE if state in ("arming", "active") else GRAY, 0.36)
        _bar(img, 182, y - 8, 70, 8, prog if state in ("arming", "active") else 0.0, color)
        note = f"cool {cool:.0f}s" if cool > 0 else s.suppressed.get(g, "")
        if note:
            _text(img, note[:15], 258, y, DIM, 0.34)
        y += 15
        if g == "eyebrows_up" and s.brow is not None:  # the eyebrow meter, right under its row
            level, base, thr = s.brow
            _text(img, " brows", 10, y + 2, GRAY, 0.36)
            _bar(img, 72, y - 6, 110, 8, level, CYAN, [(base, GRAY), (thr, RED)])
            _text(img, f"now {level:.2f}  rest {base:.2f}  need {thr:.2f}", 188, y + 2, DIM, 0.36)
            y += 16

    # effects and sound
    y += 4
    fx = ", ".join(s.effects) if s.effects else "-"
    _text(img, f"fx: {fx}"[:52], 10, y, WHITE, 0.42)
    y += 16
    extra = "  ".join(t for t in (s.outro, s.sound, s.plate) if t)
    if extra:
        _text(img, extra[:60], 10, y, GRAY, 0.38)

    # warnings, newest last, above the help line
    rows = s.warnings[-6:]
    if rows:
        top = H - 24 - 17 * len(rows)
        _shade(img, 0, top - 14, W, H - 18, 0.5)
        for i, (sev, text) in enumerate(rows):
            c = RED if sev == "error" else ORANGE if sev == "warn" else CYAN
            _text(img, text[:118], 10, top + 17 * i, c, 0.42)
    _shade(img, 0, H - 18, W, H, 0.6)
    _text(img, s.hotkeys[:150], 8, H - 5, DIM, 0.36)


def info_card(width: int, height: int, title: str, lines: list[str] = ()) -> np.ndarray:
    """A plain dark card with text, e.g. what Zoom sees while the camera reconnects."""
    img = np.full((height, width, 3), 24, np.uint8)
    k = width / 1280
    _text(img, title, int(60 * k), int(height * 0.45), WHITE, 1.4 * k, max(1, int(3 * k)))
    for i, line in enumerate(lines):
        _text(img, line, int(60 * k), int(height * 0.45 + (60 + 40 * i) * k), GRAY, 0.8 * k, max(1, int(2 * k)))
    return img
