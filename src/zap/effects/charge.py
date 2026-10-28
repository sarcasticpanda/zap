"""Ember sparks gathering at the snapping hand while the snap count is still open.

Snaps resolve only after the ~0.65 s counting window closes, so without a
pre-roll the first snap looks ignored. A few dozen sparks orbit and rise
around the hand, brightening over 0.6 s. Everything happens in one small
square around the hand: sparks (with short trails) are splatted into a float
canvas, blurred twice (tight core + wide halo), mapped through a fire palette
and added to the frame, so the glow is additive and costs well under 1 ms.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import cv2
import numpy as np

from zap.effects.base import Effect, FxContext, Layer
from zap.effects.registry import register
from zap.render.blend import clamp01, ease_out_cubic

log = logging.getLogger(__name__)

RAMP_S = 0.6  # time to reach full intensity
FADE_S = 0.15  # fade after discharge() and before the auto end
MAX_S = 1.2  # auto end
N_MIN, N_MAX = 20, 40  # sparks at the start / at full charge
TRAIL = 3  # samples per spark: head + trail
TRAIL_DT = 0.02  # seconds between trail samples
TRAIL_W = np.array([1.0, 0.55, 0.3], np.float32)
CORE_SIGMA, HALO_SIGMA = 1.2, 4.0
CORE_GAIN, HALO_GAIN = 8.0, 30.0  # one isolated spark -> ~0.9 core, ~0.3 halo
HAND_GLOW = 0.3  # peak of the soft glow over the hand at full charge
RADIUS_FRAC = 0.10  # default orbit radius as a fraction of the frame height


def _fire_lut() -> np.ndarray:
    """(256, 1, 3) BGR palette: dark red -> orange -> yellow -> white-hot."""
    x = np.arange(256, dtype=np.float32) / 255.0
    r = np.clip(2.4 * x, 0, 1)
    g = np.clip(1.8 * x - 0.35, 0, 1) * 0.92
    b = np.clip(2.5 * x - 1.5, 0, 1)
    return (np.stack([b, g, r], axis=-1) * 255).astype(np.uint8).reshape(256, 1, 3)


FIRE_LUT = _fire_lut()


@register("charge")
class Charge(Effect):
    """OVERLAY pre-roll: start(ctx, where=(x, y)); follow(where) to track the hand; discharge() to end."""

    layer = Layer.OVERLAY

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._rng = np.random.default_rng(params.get("seed"))
        self._radius_param = float(params.get("radius", 0.0))
        self._cx = self._cy = 0.0
        self._last_now = 0.0
        self._t_discharge: float | None = None
        self._side = 0

    # ------------------------------------------------------------ control
    def start(self, ctx: FxContext, **kw: Any) -> None:
        super().start(ctx, **kw)
        where = kw.get("where")
        if where is None:
            log.debug("charge started without a hand position; using the frame centre")
            where = (ctx.W / 2, ctx.H / 2)
        self._cx, self._cy = float(where[0]), float(where[1])
        self._last_now = ctx.now
        self._t_discharge = None
        r = float(kw.get("radius") or self._radius_param or RADIUS_FRAC * ctx.H)
        self._build(max(8.0, r))

    def follow(self, where: tuple[float, float] | None) -> None:
        """Move the sparks toward the hand's new position (lightly smoothed)."""
        if where is None:
            return
        self._cx += 0.5 * (float(where[0]) - self._cx)
        self._cy += 0.5 * (float(where[1]) - self._cy)

    def discharge(self) -> None:
        """The snap count resolved: fade out over 150 ms."""
        if self._t_discharge is None:
            self._t_discharge = self._last_now

    def release(self, ctx: FxContext) -> None:
        self._last_now = ctx.now
        self.discharge()

    # ------------------------------------------------------------ setup
    def _build(self, r: float) -> None:
        rng = self._rng
        n = N_MAX
        f32 = np.float32
        self._r = r
        self._theta = rng.uniform(0, 2 * math.pi, n).astype(f32)
        sign = np.where(rng.random(n) < 0.8, 1.0, -1.0)  # mostly one direction reads as a swirl
        self._omega = (rng.uniform(2.5, 5.0, n) * sign).astype(f32)
        self._r0 = (rng.uniform(0.55, 1.0, n) * r).astype(f32)
        self._life = rng.uniform(0.35, 0.7, n).astype(f32)
        self._phase = rng.random(n).astype(f32)
        self._rise = (rng.uniform(0.6, 1.3, n) * r).astype(f32)  # px per second
        self._weight = rng.uniform(0.6, 1.0, n).astype(f32)
        self._flick_f = rng.uniform(12.0, 25.0, n).astype(f32)
        self._flick_p = rng.uniform(0, 2 * math.pi, n).astype(f32)

        side = math.ceil(3.6 * r) | 1
        self._side = side
        self._ox, self._oy = side / 2.0, side * 0.62  # hand position inside the canvas (room above to rise)
        self._canvas = np.zeros((side, side), f32)
        self._core = np.empty_like(self._canvas)
        self._halo = np.empty_like(self._canvas)
        self._acc = np.empty_like(self._canvas)
        self._u8 = np.empty((side, side), np.uint8)
        self._bgr = np.empty((side, side, 3), np.uint8)
        self._col = np.empty((side, side, 3), np.uint8)
        ys = np.arange(side, dtype=f32)[:, None] - f32(self._oy)
        xs = np.arange(side, dtype=f32)[None, :] - f32(self._ox)
        self._glow = np.exp(-(xs * xs + ys * ys) / f32(2 * (0.45 * r) ** 2)).astype(f32)

    # ------------------------------------------------------------ frame
    def _level(self, now: float) -> tuple[float, float]:
        """(intensity 0..1, discharge progress 0..1); intensity 0 means finished."""
        e = now - self.t0
        if e >= MAX_S:
            return 0.0, 0.0
        level = (0.25 + 0.75 * ease_out_cubic(e / RAMP_S)) * clamp01((MAX_S - e) / FADE_S)
        burst = 0.0
        if self._t_discharge is not None:
            burst = clamp01((now - self._t_discharge) / FADE_S)
            level *= 1.0 - burst
        return level, burst

    def update(self, ctx: FxContext, out: np.ndarray) -> None:
        self._last_now = ctx.now
        if not self._side:
            self.active = False
            return
        level, burst = self._level(ctx.now)
        if level <= 0.0:
            self.active = False
            return
        e = ctx.now - self.t0
        n = N_MIN + round((N_MAX - N_MIN) * clamp01(e / RAMP_S))
        self._splat(e, n, level, 1.0 + 0.8 * burst)
        self._shade(level)
        self._add(out)

    def _splat(self, e: float, n: int, level: float, spread: float) -> None:
        f32 = np.float32
        te = np.maximum(f32(e) - np.arange(TRAIL, dtype=f32)[:, None] * f32(TRAIL_DT), f32(0.0))  # (TRAIL, 1)
        life = self._life[:n]
        u = np.mod(te / life + self._phase[:n], f32(1.0))  # (TRAIL, n) fraction of each spark's life
        ang = self._theta[:n] + self._omega[:n] * te
        rad = self._r0[:n] * (f32(1.0) - f32(0.55) * u) * f32(spread)  # spiral inward, outward on discharge
        x = f32(self._ox) + rad * np.cos(ang)
        y = f32(self._oy) + f32(0.45) * rad * np.sin(ang) - self._rise[:n] * u * life
        flick = f32(0.75) + f32(0.25) * np.sin(self._flick_f[:n] * te + self._flick_p[:n])
        w = self._weight[:n] * np.sin(f32(math.pi) * u) * flick * TRAIL_W[:, None] * f32(level)
        xi, yi = x.astype(np.int32).ravel(), y.astype(np.int32).ravel()
        s = self._side
        ok = (xi >= 0) & (xi < s) & (yi >= 0) & (yi < s)
        self._canvas.fill(0.0)
        np.add.at(self._canvas.ravel(), yi[ok] * s + xi[ok], w.ravel()[ok])

    def _shade(self, level: float) -> None:
        cv2.GaussianBlur(self._canvas, (0, 0), CORE_SIGMA, dst=self._core)
        cv2.GaussianBlur(self._canvas, (0, 0), HALO_SIGMA, dst=self._halo)
        cv2.addWeighted(self._core, CORE_GAIN * 255.0, self._halo, HALO_GAIN * 255.0, 0.0, dst=self._acc)
        cv2.scaleAdd(self._glow, HAND_GLOW * 255.0 * level, self._acc, dst=self._acc)
        cv2.convertScaleAbs(self._acc, dst=self._u8)
        cv2.cvtColor(self._u8, cv2.COLOR_GRAY2BGR, dst=self._bgr)
        cv2.LUT(self._bgr, FIRE_LUT, dst=self._col)

    def _add(self, out: np.ndarray) -> None:
        H, W = out.shape[:2]
        bx, by = round(self._cx - self._ox), round(self._cy - self._oy)
        x0, y0 = max(0, bx), max(0, by)
        x1, y1 = min(W, bx + self._side), min(H, by + self._side)
        if x1 <= x0 or y1 <= y0:
            return
        roi = out[y0:y1, x0:x1]
        cv2.add(roi, self._col[y0 - by:y1 - by, x0 - bx:x1 - bx], dst=roi)
