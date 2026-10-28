"""Absolute Cinema: raise both hands and the meeting sees you as a film still.

The look runs on ONE channel because the result is grayscale anyway: a single
cvtColor, then zoom, S-curve, background darkening, vignette, grain, flash and
even the title text all touch a third of the bytes, and the frame only becomes
three channels again in the last step. Anything that does not change per frame
(vignette, grain, the title at its pop-in scales) is built once per process on a
background thread; until it is ready the look simply skips that ingredient, so
starting the effect never stalls a frame.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from functools import partial
from typing import Any

import cv2
import numpy as np

from zap.effects.base import Effect, FxContext, Layer
from zap.effects.registry import register
from zap.render.blend import (
    clamp01,
    ease_in_out_sine,
    ease_out_cubic,
    grain_frames,
    s_curve_lut,
    vignette_map,
    zoom_about,
)
from zap.render.text import TextStyle, render_text, scale_rgba
from zap.signals import WRIST
from zap.vision.geometry import OneEuro, is_upright

log = logging.getLogger(__name__)

LAYOUTS = ("left_right", "top_bottom")
WORDS = ("ABSOLUTE", "CINEMA")
WORD_DELAYS = (0.2, 0.45)  # seconds after start each word fades in

# camera
PUNCH_S, PUNCH_SCALE = 0.6, 1.08  # smooth ease-in-out, no overshoot
ENTER_S = 0.45  # colour -> black-and-white cross-fade at the start
PUSH_S, PUSH_SCALE = 6.0, 1.16  # slow push after the punch-in
# timeline
HOLD_S = 0.5  # stays fully on this long after the hands come down
FADE_S = 0.6
MAX_S = 12.0  # fully gone by then even if the hands stay up
FLASH_S, FLASH_PEAK = 0.12, 0.0  # no flash: the user wants a smooth transition
POP_S, POP_FROM = 0.35, 1.12
POP_SCALES = tuple(float(s) for s in np.linspace(POP_FROM, 1.0, 7))
# look
BG_LEVEL = 0.72  # background brightness relative to the person (keep the room visible)
GRAIN_SIGMA, GRAIN_COUNT = 10.0, 8
VIGNETTE_STRENGTH, VIGNETTE_RADIUS = 0.55, 0.9
# title
TEXT_SIZE = 0.11  # of frame height
TRACKING = 0.06
LEVEL_RANGE = (0.35, 0.60)  # clamp for the title's vertical centre, fraction of H
LEVEL_DEFAULT = 0.45
TOP_BOTTOM_Y = (0.14, 0.86)
MAX_OCCLUSION = 0.85  # more of a word than this "behind" the person means the mask is wrong: draw in front

_SCURVE = s_curve_lut(28)
_CLAHE = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(6, 6))
# The person map stores 255 * (BG_LEVEL + (1 - BG_LEVEL) * m); text fades behind the person for m in 0.4..0.6.
_OCC_LO = int(255 * (BG_LEVEL + (1 - BG_LEVEL) * 0.4))
_OCC_HI = int(255 * (BG_LEVEL + (1 - BG_LEVEL) * 0.6))


# ---------------------------------------------------------------- one-time builds
class _Prebuilt:
    """Builds expensive things once per process on daemon threads; ``get`` never blocks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._done: dict[tuple, Any] = {}
        self._events: dict[tuple, threading.Event] = {}

    def get(self, key: tuple, build: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._done:
                return self._done[key]
            if key not in self._events:
                ev = self._events[key] = threading.Event()
                threading.Thread(target=self._run, args=(key, build, ev), name="zap-cinema-prebuild", daemon=True).start()
        return None

    def wait(self, key: tuple, timeout: float) -> bool:
        with self._lock:
            ev = self._events.get(key)
        return ev is not None and ev.wait(timeout)

    def _run(self, key: tuple, build: Callable[[], Any], ev: threading.Event) -> None:
        try:
            value = build()
        except Exception:
            log.exception("cinema: building %s failed; the look continues without it", key[0])
            value = None
        with self._lock:
            self._done[key] = value
        ev.set()


_prebuilt = _Prebuilt()


class _Maps:
    def __init__(self, w: int, h: int) -> None:
        self.vignette = np.ascontiguousarray(vignette_map(w, h, VIGNETTE_STRENGTH, VIGNETTE_RADIUS)[..., 0])
        self.grain_pos: list[np.ndarray] = []
        self.grain_neg: list[np.ndarray] = []
        # Signed noise split into two saturating uint8 passes: cv2.add then cv2.subtract, no int16 math per frame.
        for n in grain_frames(w, h, sigma=GRAIN_SIGMA, count=GRAIN_COUNT, seed=11):
            self.grain_pos.append(np.clip(n, 0, 255).astype(np.uint8))
            self.grain_neg.append(np.clip(-n, 0, 255).astype(np.uint8))


def _gray_premult(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RGBA -> (premultiplied gray, inverse alpha), both single-channel uint8."""
    lum = cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]), cv2.COLOR_RGB2GRAY)
    a = np.ascontiguousarray(rgba[..., 3])
    return cv2.multiply(lum, a, scale=1 / 255), 255 - a


def _build_word(text: str, font: str, size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    style = TextStyle(font=font, size=size, tracking=TRACKING, shadow=max(2, size // 12), shadow_alpha=0.6)
    rgba = render_text(text, style)
    return [_gray_premult(scale_rgba(rgba, s)) for s in POP_SCALES]


def _maps_key(w: int, h: int) -> tuple:
    return ("maps", w, h)


def _word_key(word: str, font: str, h: int) -> tuple:
    return ("word", word, font, round(TEXT_SIZE * h))


def prewarm(width: int = 1280, height: int = 720, font: str = "timesbd.ttf") -> None:
    """Start building the grain/vignette maps and title sprites for this output size (non-blocking)."""
    _prebuilt.get(_maps_key(width, height), partial(_Maps, width, height))
    for word in WORDS:
        key = _word_key(word, font, height)
        _prebuilt.get(key, partial(_build_word, word, font, key[3]))


def wait_ready(width: int = 1280, height: int = 720, font: str = "timesbd.ttf", timeout: float = 10.0) -> bool:
    """Block until prewarm() for this size has finished (for tests and app start-up)."""
    prewarm(width, height, font)
    keys = [_maps_key(width, height), *(_word_key(w, font, height) for w in WORDS)]
    return all(_prebuilt.wait(k, timeout) for k in keys)


# ---------------------------------------------------------------- helpers
def _crop_origin(scale: float, cx: float, cy: float, w: int, h: int) -> tuple[float, float, float]:
    """(effective scale, x0, y0) of the crop zoom_about uses, so points can follow the zoom."""
    if scale <= 1.001:
        return 1.0, 0.0, 0.0
    cw, ch = w / scale, h / scale
    return scale, min(max(cx - cw / 2, 0.0), w - cw), min(max(cy - ch / 2, 0.0), h - ch)


def _blit_gray(
    dst: np.ndarray,
    pm: np.ndarray,
    inv: np.ndarray,
    cx: float,
    cy: float,
    alpha: float = 1.0,
    person: np.ndarray | None = None,
) -> None:
    """Composite a single-channel premultiplied sprite centred at (cx, cy), optionally behind the person."""
    H, W = dst.shape[:2]
    h, w = pm.shape[:2]
    x, y = round(cx - w / 2), round(cy - h / 2)
    dx0, dy0 = max(0, x), max(0, y)
    dx1, dy1 = min(W, x + w), min(H, y + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    sx0, sy0 = dx0 - x, dy0 - y
    p = pm[sy0 : sy0 + dy1 - dy0, sx0 : sx0 + dx1 - dx0]
    iv = inv[sy0 : sy0 + dy1 - dy0, sx0 : sx0 + dx1 - dx0]
    if alpha < 0.999:
        p = cv2.convertScaleAbs(p, alpha=alpha)
        iv = cv2.convertScaleAbs(iv, alpha=alpha, beta=255 * (1 - alpha))
    if person is not None:
        occ = cv2.convertScaleAbs(cv2.subtract(person[dy0:dy1, dx0:dx1], _OCC_LO), alpha=255 / (_OCC_HI - _OCC_LO))
        if cv2.mean(occ)[0] < MAX_OCCLUSION * 255:
            keep = 255 - occ
            p = cv2.multiply(p, keep, scale=1 / 255)
            iv = 255 - cv2.multiply(255 - iv, keep, scale=1 / 255)
    roi = dst[dy0:dy1, dx0:dx1]
    cv2.multiply(roi, iv, dst=roi, scale=1 / 255)
    cv2.add(roi, p, dst=roi)


def camera_scale(t: float) -> float:
    """Zoom factor t seconds into the effect: punch-in with overshoot, then a slow push."""
    if t < PUNCH_S:
        return 1.0 + (PUNCH_SCALE - 1.0) * ease_in_out_sine(t / PUNCH_S)
    return PUNCH_SCALE + (PUSH_SCALE - PUNCH_SCALE) * clamp01((t - PUNCH_S) / PUSH_S)


# ---------------------------------------------------------------- effect
@register("cinema")
class Cinema(Effect):
    """Hold effect: on while both hands are up; release() -> 0.5 s hold -> 0.3 s fade."""

    layer = Layer.LOOK
    needs_mask = True
    needs_face = True
    sfx_key = "cinema"

    def __init__(self, layout: str = "left_right", font: str = "timesbd.ttf", text_behind: bool = False, **params: Any):
        super().__init__(layout=layout, font=font, text_behind=text_behind, **params)
        if layout not in LAYOUTS:
            log.warning("cinema: unknown layout %r, using left_right", layout)
            layout = "left_right"
        self.layout = layout
        self.font = font
        self.text_behind = text_behind
        self._release_at: float | None = None
        self._focus_filter = OneEuro(min_cutoff=0.5, beta=0.003)
        self._level_filter = OneEuro(min_cutoff=0.12, beta=0.0)  # words settle and stay put
        self._focus_target: tuple[float, float] | None = None
        self._frame_no = 0
        self._bufs: dict[str, np.ndarray] = {}
        prewarm(1280, 720, font)

    # ------------------------------------------------------------ control
    def start(self, ctx: FxContext, **kw: Any) -> None:
        if self.active and (self._release_at is None or ctx.now - self._release_at < HOLD_S):
            self._release_at = None  # hands went back up before the fade began: just keep going
            return
        super().start(ctx, **kw)
        self._release_at = None
        self._focus_filter.reset()
        self._level_filter.reset()
        self._focus_target = None
        self._frame_no = 0
        prewarm(ctx.W, ctx.H, self.font)

    def release(self, ctx: FxContext) -> None:
        if self.active and self._release_at is None:
            self._release_at = ctx.now

    @property
    def released(self) -> bool:
        return self._release_at is not None

    # ------------------------------------------------------------ frame
    def update(self, ctx: FxContext, out: np.ndarray) -> None:
        H, W = out.shape[:2]
        t = ctx.now - self.t0
        if self._release_at is None and t >= MAX_S - HOLD_S - FADE_S:
            self._release_at = self.t0 + MAX_S - HOLD_S - FADE_S
        mix = ease_in_out_sine(t / ENTER_S) if t < ENTER_S else 1.0  # 1 = full look, 0 = untouched frame
        scale = camera_scale(t)
        if self._release_at is not None:
            tr = ctx.now - self._release_at
            if tr >= HOLD_S + FADE_S:
                self.active = False
                return
            if tr > HOLD_S:
                mix = min(mix, 1.0 - ease_in_out_sine((tr - HOLD_S) / FADE_S))
                scale = 1.0 + (scale - 1.0) * mix  # zoom eases out with the fade so nothing ghosts

        cx, cy = self._focus(ctx, W, H)
        scale, x0, y0 = _crop_origin(scale, cx, cy, W, H)
        gray = self._buf("gray", (H, W))
        look = self._buf("look", (H, W))
        cv2.cvtColor(out, cv2.COLOR_BGR2GRAY, dst=gray)
        zoom_about(gray, scale, cx, cy, out=look)
        _CLAHE.apply(look, dst=look)  # bring a backlit face out of the shadows
        cv2.LUT(look, _SCURVE, dst=look)

        person = self._person_map(ctx, scale, x0, y0, W, H)
        if person is not None:
            cv2.multiply(look, person, dst=look, scale=1 / 255)
        maps = _prebuilt.get(_maps_key(W, H), partial(_Maps, W, H))
        if maps is not None:
            cv2.multiply(look, maps.vignette, dst=look, scale=1 / 255)
            k = self._frame_no % len(maps.grain_pos)
            cv2.add(look, maps.grain_pos[k], dst=look)
            cv2.subtract(look, maps.grain_neg[k], dst=look)
        if t < FLASH_S:
            a = FLASH_PEAK * (1.0 - t / FLASH_S)
            cv2.convertScaleAbs(look, dst=look, alpha=1.0 - a, beta=255.0 * a)
        self._draw_words(ctx, look, t, person if self.text_behind else None, scale, y0)

        if mix >= 1.0:
            cv2.cvtColor(look, cv2.COLOR_GRAY2BGR, dst=out)
        else:
            look3 = self._buf("look3", (H, W, 3))
            color = self._buf("color", (H, W, 3))
            cv2.cvtColor(look, cv2.COLOR_GRAY2BGR, dst=look3)
            zoom_about(out, scale, cx, cy, out=color)
            cv2.addWeighted(look3, mix, color, 1.0 - mix, 0.0, dst=out)
        self._frame_no += 1

    # ------------------------------------------------------------ pieces
    def _buf(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        b = self._bufs.get(name)
        if b is None or b.shape != shape:
            b = self._bufs[name] = np.empty(shape, np.uint8)
        return b

    def _focus(self, ctx: FxContext, W: int, H: int) -> tuple[float, float]:
        """Smoothed zoom centre: the face if seen (or last seen), else the frame centre."""
        face = ctx.sig.face
        if face is not None:
            c = face.center
            self._focus_target = (float(c[0]), float(c[1]))
        target = self._focus_target or (W / 2, H / 2)
        f = self._focus_filter(np.asarray(target), ctx.now)
        return float(f[0]), float(f[1])

    def _person_map(self, ctx: FxContext, scale: float, x0: float, y0: float, W: int, H: int) -> np.ndarray | None:
        """uint8 brightness multiplier in zoomed output coords: 255 on the person, BG_LEVEL*255 behind."""
        mask = ctx.sig.mask
        if mask is None:
            return None
        m = mask[..., 0] if mask.ndim == 3 else mask
        mh, mw = m.shape[:2]
        low = cv2.convertScaleAbs(m, alpha=255 * (1 - BG_LEVEL), beta=255 * BG_LEVEL)
        kx, ky = W / mw, H / mh
        # low-res pixel centre i sits at (i + 0.5) * k - 0.5 in the frame; then the zoom maps p -> (p - x0) * scale
        M = np.float32(
            [[scale * kx, 0, scale * (0.5 * kx - 0.5 - x0)], [0, scale * ky, scale * (0.5 * ky - 0.5 - y0)]]
        )
        dst = self._buf("person", (H, W))
        cv2.warpAffine(low, M, (W, H), dst=dst, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return dst

    def _word_centres(self, ctx: FxContext, W: int, H: int, scale: float, y0: float) -> list[tuple[float, float]]:
        if self.layout == "top_bottom":
            return [(0.5 * W, TOP_BOTTOM_Y[0] * H), (0.5 * W, TOP_BOTTOM_Y[1] * H)]
        hands = ctx.sig.hands
        raised = [h for h in hands if is_upright(h)] or hands
        if raised:
            wrist_y = float(np.mean([h.lm[WRIST, 1] for h in raised]))
            target = (wrist_y - y0) * scale  # follow the hands through the zoom
        else:
            target = LEVEL_DEFAULT * H
        target = min(max(target, LEVEL_RANGE[0] * H), LEVEL_RANGE[1] * H)
        y = float(self._level_filter(target, ctx.now))
        return [(0.2 * W, y), (0.8 * W, y)]

    def _draw_words(
        self, ctx: FxContext, look: np.ndarray, t: float, person: np.ndarray | None, scale: float, y0: float
    ) -> None:
        H, W = look.shape[:2]
        centres = self._word_centres(ctx, W, H, scale, y0)
        for word, delay, (cx, cy) in zip(WORDS, WORD_DELAYS, centres):
            p = (t - delay) / POP_S
            if p <= 0:
                continue
            key = _word_key(word, self.font, H)
            levels = _prebuilt.get(key, partial(_build_word, word, self.font, key[3]))
            if levels is None:
                continue
            e = ease_out_cubic(p)
            s = POP_FROM + (1.0 - POP_FROM) * e
            idx = min(range(len(POP_SCALES)), key=lambda i: abs(POP_SCALES[i] - s))
            pm, inv = levels[idx]
            _blit_gray(look, pm, inv, cx, cy, e, person)
