"""Money rain: three snaps and the meeting gets paid.

Bills are particles in numpy arrays, so the per-frame maths is a handful of
vectorised ops no matter how many are falling. Every pose a bill can take
(21 tilts x 6 flip phases x 2 depths per sprite variant) is pre-rendered once
per process on a background thread as premultiplied BGR, so drawing one is two
cv2 calls on its own rectangle. A small subset is published first, and bills
spawn above the frame, so starting the effect never waits for the atlas.

About a quarter of the bills fall *behind* you: after drawing one, the person
is put back over it with the segmentation mask, inside that bill's rectangle
only, and only when the low-res mask says the person is actually there.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from zap.effects.base import Effect, FxContext, Layer
from zap.effects.registry import register
from zap.render.blend import blit_premult, clamp01, ease_in_out_sine, make_premult, shake_offset
from zap.render.text import TextStyle, load_font, render_emoji, render_text

log = logging.getLogger(__name__)

# particles
CAP = 160
BURST = 50  # spawned on the first frame, staggered above the top edge
RATE = 30.0  # bills per second after the burst ...
SPAWN_S = 3.5  # ... until this long into the effect
CLEAR_BY_S = 6.0  # every bill falls fast enough to be gone by about then
MAX_S = 10.0  # hard stop, whatever happens
BEHIND_RATIO = 0.25
NEAR_RATIO = 0.6  # of the bills in front
SPEED = ((160.0, 230.0), (220.0, 300.0))  # px/s at 720p for (far, near)
SWAY_PX = (15.0, 55.0)  # at 720p
SWAY_HZ = (0.35, 0.9)
TILT_DEG = (15.0, 50.0)

# atlas
TILTS = np.arange(-50, 51, 5)  # degrees, cv2 convention (positive = counter-clockwise on screen)
N_FLIP = 6  # phases of one full tumble; the back face shows while cos(phase) < 0
MIN_SQUASH = 0.1
SUBSET_TILTS = (2, 6, 10, 14, 18)  # indices into TILTS for the quick first atlas
BILL_ASPECT = 2.35
BILL_W = (0.15, 0.205)  # bill width as a fraction of frame height, (far, near)
EMOJI_SIZE = (0.075, 0.10)
BILL_GREEN = "#85BB65"
BILL_BACK = "#6A9A52"
BILL_BORDER = "#4F7942"
BILL_INK = "#2F4A27"


@dataclass(frozen=True)
class _Variant:
    kind: str  # "bill" or a single emoji codepoint
    weight: float
    flips: bool  # tumbles (front/back) or only tilts


VARIANTS = (
    _Variant("bill", 0.70, True),
    _Variant("\U0001f4b5", 0.12, True),  # banknote with dollar sign
    _Variant("\U0001f4b8", 0.10, False),  # money with wings
    _Variant("\U0001f4b0", 0.08, False),  # money bag
)
_WEIGHTS = np.array([v.weight for v in VARIANTS]) / sum(v.weight for v in VARIANTS)

# stamp
STAMP_T0 = 0.35  # seconds into the effect
SLAM_S = 0.15
STAMP_FROM = 3.0
STAMP_HOLD_S = 1.5
STAMP_FADE_S = 0.35
SHAKE_PX = 4.0
SHAKE_DECAY = 7.0
STAMP_ANGLE = 12.0  # degrees counter-clockwise on screen (CSS rotate(-12deg)): text rises to the right
STAMP_RED = (206, 28, 36, 255)
STAMP_POS = (0.5, 0.62)  # centre as a fraction of (W, H)
STAMP_TEXT_SIZE = 0.13
CAPTION_SIZE = 0.065
CAPTION_FADE_S = 0.2


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
                threading.Thread(target=self._run, args=(key, build, ev), name="zap-money-prebuild", daemon=True).start()
        return None

    def wait(self, key: tuple, timeout: float) -> bool:
        with self._lock:
            ev = self._events.get(key)
        return ev is not None and ev.wait(timeout)

    def _run(self, key: tuple, build: Callable[[], Any], ev: threading.Event) -> None:
        try:
            value = build()
        except Exception:
            log.exception("money: building %s failed", key[0])
            value = None
        with self._lock:
            self._done[key] = value
        ev.set()


_prebuilt = _Prebuilt()
_quick_atlas: dict[int, _Atlas] = {}  # height -> subset atlas, published before the full one


def _bill_rgba(w: int, h: int, back: bool) -> np.ndarray:
    """A procedural dollar bill (RGBA), drawn 3x supersampled for clean edges."""
    ss = 3
    W, H = w * ss, h * ss
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    border = max(2, int(0.07 * H))
    d.rounded_rectangle((0, 0, W - 1, H - 1), radius=int(0.08 * H), fill=BILL_BACK if back else BILL_GREEN,
                        outline=BILL_BORDER, width=border)
    inset = int(0.15 * H)
    d.rectangle((inset, inset, W - 1 - inset, H - 1 - inset), outline=BILL_INK, width=max(1, int(0.025 * H)))
    cx, cy = W / 2, H / 2
    if back:
        bw, bh = 0.5 * H, 0.26 * H  # a little building with columns
        d.rounded_rectangle((cx - bw, cy - bh, cx + bw, cy + bh), radius=int(0.05 * H), fill="#4E7A3E",
                            outline=BILL_INK, width=max(1, int(0.02 * H)))
        for i in range(1, 7):
            x = cx - bw + i * 2 * bw / 7
            d.line((x, cy - 0.7 * bh, x, cy + 0.7 * bh), fill="#9CC27F", width=max(1, int(0.02 * H)))
    else:
        rx, ry = 0.21 * H, 0.33 * H  # portrait oval with a head-and-shoulders silhouette
        d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=BILL_INK, outline="#1E3318", width=max(1, int(0.02 * H)))
        d.ellipse((cx - 0.09 * H, cy - 0.2 * H, cx + 0.09 * H, cy + 0.02 * H), fill="#4F7A3F")
        d.chord((cx - 0.17 * H, cy + 0.03 * H, cx + 0.17 * H, cy + 0.4 * H), 180, 360, fill="#4F7A3F")
        for sx in (0.27 * W, 0.73 * W):  # treasury seals
            r = 0.12 * H
            d.ellipse((sx - r, cy - r, sx + r, cy + r), outline=BILL_INK, width=max(1, int(0.025 * H)))
    font = load_font("georgiab.ttf", max(6, int(0.24 * H)))
    pad = inset + int(0.04 * H)
    for (x, y), anchor in (((pad, pad), "lt"), ((W - pad, pad), "rt"), ((pad, H - pad), "lb"), ((W - pad, H - pad), "rb")):
        d.text((x, y), "100", font=font, fill=BILL_INK, anchor=anchor)
    return np.asarray(img.resize((w, h), Image.LANCZOS)).copy()


def _faces(variant: _Variant, depth: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """(front, back) RGBA for one variant at one depth."""
    if variant.kind == "bill":
        w = max(8, round(BILL_W[depth] * height))
        h = max(4, round(w / BILL_ASPECT))
        return _bill_rgba(w, h, back=False), _bill_rgba(w, h, back=True)
    front = render_emoji(variant.kind, max(8, round(EMOJI_SIZE[depth] * height)))
    back = np.ascontiguousarray(front[:, ::-1])
    back[..., :3] = (back[..., :3] * 0.6).astype(np.uint8)  # seen from behind: mirrored and in shadow
    return front, back


def _pose(pm: np.ndarray, inv: np.ndarray, squash: float, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Squash (fake tumble) and rotate a premultiplied sprite onto a canvas that fits it."""
    h, w = pm.shape[:2]
    if squash < 0.999:
        w = max(2, round(w * squash))
        pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_AREA)
        inv = cv2.resize(inv, (w, h), interpolation=cv2.INTER_AREA)
    c, s = abs(math.cos(math.radians(angle))), abs(math.sin(math.radians(angle)))
    bw, bh = math.ceil(w * c + h * s) + 2, math.ceil(w * s + h * c) + 2
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[0, 2] += bw / 2 - w / 2
    M[1, 2] += bh / 2 - h / 2
    pm_r = cv2.warpAffine(pm, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    inv_r = cv2.warpAffine(inv, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255, 255, 255))
    return pm_r, inv_r


@dataclass
class _Atlas:
    pm: list[np.ndarray]
    inv: list[np.ndarray]
    half: np.ndarray  # (n, 2) int32: half width, half height of each sprite
    lut: np.ndarray  # (variants, 2 depths, tilts, flips) int32 -> sprite index
    complete: bool


def _flip_cos(fi: int) -> float:
    return math.cos(2 * math.pi * fi / N_FLIP)


def _build_atlas(height: int, complete: bool) -> _Atlas:
    variants = range(len(VARIANTS)) if complete else (0,)
    tilt_ids = range(len(TILTS)) if complete else SUBSET_TILTS
    flip_ids = range(N_FLIP) if complete else (0, N_FLIP // 2)
    pm: list[np.ndarray] = []
    inv: list[np.ndarray] = []
    half: list[tuple[int, int]] = []
    built: dict[tuple[int, int, int, int], int] = {}
    for v in variants:
        var = VARIANTS[v]
        for depth in (0, 1):
            front, back = _faces(var, depth, height)
            sides = {True: make_premult(front), False: make_premult(back)}
            same: dict[tuple[bool, float, int], int] = {}  # phases k and N-k look identical: build once
            for fi in flip_ids if var.flips else (0,):
                c = _flip_cos(fi)
                side, squash = c >= -1e-6, max(abs(c), MIN_SQUASH)
                for ti in tilt_ids:
                    key = (side, round(squash, 4), ti)
                    if key not in same:
                        p, i = _pose(*sides[side], squash, float(TILTS[ti]))
                        same[key] = len(pm)
                        pm.append(p)
                        inv.append(i)
                        half.append((p.shape[1] // 2, p.shape[0] // 2))
                    built[(v, depth, ti, fi)] = same[key]
    # Every (variant, depth, tilt, flip) points at the nearest pose that was actually built.
    lut = np.zeros((len(VARIANTS), 2, len(TILTS), N_FLIP), np.int32)
    for v in range(len(VARIANTS)):
        bv = v if v in variants else 0
        flips = flip_ids if VARIANTS[bv].flips else (0,)
        for depth in (0, 1):
            for ti in range(len(TILTS)):
                bt = min(tilt_ids, key=lambda k: abs(int(TILTS[k]) - int(TILTS[ti])))
                for fi in range(N_FLIP):
                    bf = min(flips, key=lambda k: abs(_flip_cos(k) - _flip_cos(fi)))
                    lut[v, depth, ti, fi] = built[(bv, depth, bt, bf)]
    return _Atlas(pm, inv, np.array(half, np.int32).reshape(-1, 2), lut, complete)


def _build_atlases(height: int) -> _Atlas:
    _quick_atlas[height] = _build_atlas(height, complete=False)
    return _build_atlas(height, complete=True)


@dataclass
class _Stamp:
    pm: np.ndarray
    inv: np.ndarray
    caption: tuple[np.ndarray, np.ndarray] | None


def _build_stamp(text: str, caption: str, height: int) -> _Stamp | None:
    if not text and not caption:
        return None
    size = max(12, round(STAMP_TEXT_SIZE * height))
    pm = inv = np.zeros((0, 0, 3), np.uint8)
    if text:
        glyphs = render_text(text, TextStyle(font="impact.ttf", size=size, color=STAMP_RED, tracking=0.03))
        gh, gw = glyphs.shape[:2]
        pad, thick = int(0.28 * size), max(3, int(0.09 * size))
        W, H = gw + 2 * pad, gh + 2 * pad
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, W - 1, H - 1), radius=int(0.18 * size), outline=STAMP_RED, width=thick)
        inner = thick + max(2, thick // 2)
        d.rounded_rectangle((inner, inner, W - 1 - inner, H - 1 - inner), radius=int(0.12 * size), outline=STAMP_RED,
                            width=max(1, thick // 3))
        img.alpha_composite(Image.fromarray(glyphs), (pad, pad))
        # worn rubber-stamp ink: knock random speckles out of the alpha
        rng = np.random.default_rng(3)
        noise = cv2.resize(rng.random((max(1, H // 6), max(1, W // 6))).astype(np.float32), (W, H))
        rgba = np.asarray(img).copy()
        wear = np.where(noise > 0.78, 0.35, 0.95)
        rgba[..., 3] = (rgba[..., 3] * wear).astype(np.uint8)
        rotated = Image.fromarray(rgba).rotate(STAMP_ANGLE, resample=Image.BICUBIC, expand=True)
        pm, inv = make_premult(np.asarray(rotated))
    cap = None
    if caption:
        cs = max(10, round(CAPTION_SIZE * height))
        style = TextStyle(font="impact.ttf", size=cs, stroke=max(2, cs // 14), shadow=max(2, cs // 10), shadow_alpha=0.6)
        cap = make_premult(render_text(caption, style))
    return _Stamp(pm, inv, cap)


def prewarm(height: int = 720, stamp_text: str = "DEAL CLOSED", caption: str = "") -> None:
    """Start building the bill atlas and stamp for this output height (non-blocking)."""
    _prebuilt.get(("atlas", height), partial(_build_atlases, height))
    _prebuilt.get(("stamp", stamp_text, caption, height), partial(_build_stamp, stamp_text, caption, height))


def wait_ready(height: int = 720, stamp_text: str = "DEAL CLOSED", caption: str = "", timeout: float = 20.0) -> bool:
    """Block until prewarm() for this height has finished (for tests and app start-up)."""
    prewarm(height, stamp_text, caption)
    return _prebuilt.wait(("atlas", height), timeout) and _prebuilt.wait(
        ("stamp", stamp_text, caption, height), timeout
    )


def _atlas(height: int) -> _Atlas | None:
    full = _prebuilt.get(("atlas", height), partial(_build_atlases, height))
    return full if full is not None else _quick_atlas.get(height)


# ---------------------------------------------------------------- drawing helpers
def _shift_inplace(img: np.ndarray, dx: int, dy: int) -> None:
    """Move the picture by (dx, dy); the uncovered strip keeps its old pixels (fine for a few px of shake)."""
    if dx == 0 and dy == 0:
        return
    H, W = img.shape[:2]
    src = img[max(0, -dy) : H - max(0, dy), max(0, -dx) : W - max(0, dx)]
    img[max(0, dy) : H - max(0, -dy), max(0, dx) : W - max(0, -dx)] = src  # numpy buffers the overlap


def _fade_sprite(pm: np.ndarray, inv: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    if alpha >= 0.999:
        return pm, inv
    return cv2.convertScaleAbs(pm, alpha=alpha), cv2.convertScaleAbs(inv, alpha=alpha, beta=255 * (1 - alpha))


def _blit_scaled(out: np.ndarray, pm: np.ndarray, inv: np.ndarray, s: float, cx: float, cy: float,
                 alpha: float) -> None:
    """Blit a premultiplied sprite scaled by s about its centre, resampling only the part inside the frame."""
    H, W = out.shape[:2]
    h, w = pm.shape[:2]
    x0, y0 = cx - w * s / 2, cy - h * s / 2
    rx0, ry0 = max(0, math.floor(x0)), max(0, math.floor(y0))
    rx1, ry1 = min(W, math.ceil(x0 + w * s)), min(H, math.ceil(y0 + h * s))
    if rx1 <= rx0 or ry1 <= ry0:
        return
    M = np.float32([[s, 0, x0 - rx0], [0, s, y0 - ry0]])
    size = (rx1 - rx0, ry1 - ry0)
    p = cv2.warpAffine(pm, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    i = cv2.warpAffine(inv, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=(255, 255, 255))
    p, i = _fade_sprite(p, i, alpha)
    blit_premult(out, p, i, rx0, ry0)


# ---------------------------------------------------------------- effect
class _Particles:
    """Structure-of-arrays particle state; positions are closed-form in time since spawn."""

    def __init__(self, cap: int) -> None:
        self.alive = np.zeros(cap, bool)
        self.ts = np.zeros(cap)  # spawn time, seconds since effect start
        self.x0 = np.zeros(cap)
        self.y0 = np.zeros(cap)
        self.vy = np.zeros(cap)
        self.amp = np.zeros(cap)
        self.freq = np.zeros(cap)
        self.phi = np.zeros(cap)
        self.tilt = np.zeros(cap)
        self.flip_rate = np.zeros(cap)  # tumbles per second (signed)
        self.flip_phi = np.zeros(cap)  # in tumbles
        self.var = np.zeros(cap, np.int32)
        self.depth = np.zeros(cap, np.int32)
        self.behind = np.zeros(cap, bool)


@register("money")
class MoneyRain(Effect):
    """Fire-and-forget overlay: ends by itself once every bill has fallen out of frame (~6 s)."""

    layer = Layer.OVERLAY
    needs_mask = True  # some bills fall behind the person
    sfx_key = "money"

    def __init__(self, caption: str = "", stamp_text: str = "DEAL CLOSED", behind_ratio: float = BEHIND_RATIO,
                 seed: int | None = None, **params: Any) -> None:
        super().__init__(caption=caption, stamp_text=stamp_text, behind_ratio=behind_ratio, seed=seed, **params)
        self.caption = caption or ""
        self.stamp_text = stamp_text or ""
        self.behind_ratio = float(behind_ratio)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._p = _Particles(CAP)
        self._emitted = 0
        self._dropped = 0
        prewarm(720, self.stamp_text, self.caption)

    # ------------------------------------------------------------ control
    def start(self, ctx: FxContext, **kw: Any) -> None:
        super().start(ctx, **kw)
        self._rng = np.random.default_rng(self._seed)
        self._p = _Particles(CAP)
        self._emitted = 0
        self._dropped = 0
        prewarm(ctx.H, self.stamp_text, self.caption)

    @property
    def alive_count(self) -> int:
        return int(self._p.alive.sum())

    # ------------------------------------------------------------ frame
    def update(self, ctx: FxContext, out: np.ndarray) -> None:
        H, W = out.shape[:2]
        t = ctx.now - self.t0
        target = BURST + int(RATE * clamp01(t / SPAWN_S) * SPAWN_S)
        if target > self._emitted:
            self._spawn(target - self._emitted, t, W, H, burst=self._emitted == 0)

        stamp = _prebuilt.get(("stamp", self.stamp_text, self.caption, H),
                              partial(_build_stamp, self.stamp_text, self.caption, H))
        after_slam = t - STAMP_T0 - SLAM_S
        if stamp is not None and stamp.pm.size and 0 <= after_slam < 1.0:
            _shift_inplace(out, *shake_offset(after_slam, SHAKE_PX, SHAKE_DECAY))

        p = self._p
        idx = np.flatnonzero(p.alive)
        if idx.size:
            self._draw_bills(ctx, out, idx, t, W, H, stamp)
        else:
            self._draw_stamp(out, t, stamp)

        stamp_done = stamp is None or t >= STAMP_T0 + SLAM_S + STAMP_HOLD_S + STAMP_FADE_S
        if (t >= SPAWN_S and not p.alive.any() and stamp_done) or t >= MAX_S:
            if t >= MAX_S and p.alive.any():
                log.debug("money: hard stop with %d bills still falling", int(p.alive.sum()))
            self.active = False

    def _spawn(self, n: int, t: float, W: int, H: int, burst: bool) -> None:
        p, rng = self._p, self._rng
        self._emitted += n
        free = np.flatnonzero(~p.alive)[:n]
        k = free.size
        self._dropped += n - k
        if k == 0:
            return
        behind = rng.random(k) < self.behind_ratio
        depth = np.where(behind, 0, (rng.random(k) < NEAR_RATIO).astype(np.int32))
        reach = _reach(depth, H)
        ky = H / 720.0
        y0 = -reach - rng.random(k) * (0.5 * H if burst else 0.06 * H)
        lo = np.where(depth == 1, SPEED[1][0], SPEED[0][0])
        hi = np.where(depth == 1, SPEED[1][1], SPEED[0][1])
        vy = rng.uniform(lo, hi) * ky
        vy = np.maximum(vy, (H + reach - y0) / max(CLEAR_BY_S - t, 1.0))  # out of frame by CLEAR_BY_S
        tumble = rng.random(k) < 0.7
        p.alive[free] = True
        p.ts[free] = t
        p.x0[free] = rng.uniform(-0.03 * W, 1.03 * W, k)
        p.y0[free] = y0
        p.vy[free] = vy
        p.amp[free] = rng.uniform(*SWAY_PX, k) * ky * np.where(depth == 1, 1.0, 0.7)
        p.freq[free] = rng.uniform(*SWAY_HZ, k)
        p.phi[free] = rng.uniform(0, 2 * math.pi, k)
        p.tilt[free] = rng.uniform(*TILT_DEG, k)
        p.flip_rate[free] = np.where(tumble, rng.uniform(0.6, 2.0, k), rng.uniform(0.05, 0.3, k)) * rng.choice(
            (-1.0, 1.0), k
        )
        p.flip_phi[free] = rng.random(k)
        p.var[free] = rng.choice(len(VARIANTS), size=k, p=_WEIGHTS)
        p.depth[free] = depth
        p.behind[free] = behind

    def _draw_bills(self, ctx: FxContext, out: np.ndarray, idx: np.ndarray, t: float, W: int, H: int,
                    stamp: _Stamp | None) -> None:
        p = self._p
        depth = p.depth[idx]
        dt = t - p.ts[idx]
        ph = 2 * math.pi * p.freq[idx] * dt + p.phi[idx]
        x = p.x0[idx] + p.amp[idx] * np.sin(ph)
        y = p.y0[idx] + p.vy[idx] * dt
        reach = _reach(depth, H)
        gone = y - reach > H
        if gone.any():
            p.alive[idx[gone]] = False
        visible = ~gone & (y + reach > 0)

        atlas = _atlas(H)
        if atlas is None or not visible.any():
            self._draw_stamp(out, t, stamp)
            return
        sel = np.flatnonzero(visible)
        idx, x, y, ph, dt, depth = idx[sel], x[sel], y[sel], ph[sel], dt[sel], depth[sel]
        tilt = -p.tilt[idx] * np.cos(ph)  # lags the sway by a quarter turn: the leading edge dips, like a leaf
        ti = np.clip(np.rint((tilt - TILTS[0]) / 5.0), 0, len(TILTS) - 1).astype(np.intp)
        fi = np.floor((p.flip_phi[idx] + p.flip_rate[idx] * dt) * N_FLIP).astype(np.intp) % N_FLIP
        flat = np.ravel_multi_index((p.var[idx], depth, ti, fi), atlas.lut.shape)
        sid = np.take(atlas.lut.ravel(), flat)
        half = np.take(atlas.half, sid, axis=0)
        xs = np.rint(x - half[:, 0]).astype(np.int32)
        ys = np.rint(y - half[:, 1]).astype(np.int32)

        behind = p.behind[idx]
        # far-to-near: behind bills, front far bills, the stamp, front near bills
        groups = (np.flatnonzero(behind), np.flatnonzero(~behind & (depth == 0)), np.flatnonzero(~behind & (depth == 1)))
        pms, invs = atlas.pm, atlas.inv
        if groups[0].size:
            self._draw_behind(ctx, out, groups[0], sid, xs, ys, pms, invs)
        for j in groups[1]:
            s = sid[j]
            blit_premult(out, pms[s], invs[s], int(xs[j]), int(ys[j]))
        self._draw_stamp(out, t, stamp)
        for j in groups[2]:
            s = sid[j]
            blit_premult(out, pms[s], invs[s], int(xs[j]), int(ys[j]))

    def _draw_behind(self, ctx: FxContext, out: np.ndarray, order: np.ndarray, sid: np.ndarray, xs: np.ndarray,
                     ys: np.ndarray, pms: list[np.ndarray], invs: list[np.ndarray]) -> None:
        H, W = out.shape[:2]
        low = ctx.sig.mask
        if low is None:  # no mask: they simply fall in front
            for j in order:
                s = sid[j]
                blit_premult(out, pms[s], invs[s], int(xs[j]), int(ys[j]))
            return
        low = np.asarray(low[..., 0] if low.ndim == 3 else low, np.float32)
        mh, mw = low.shape[:2]
        low8 = cv2.convertScaleAbs(low, alpha=255)
        kx, ky = mw / W, mh / H
        for j in order:
            s = sid[j]
            pm, inv = pms[s], invs[s]
            x, y = int(xs[j]), int(ys[j])
            h, w = pm.shape[:2]
            x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
            if x1 <= x0 or y1 <= y0:
                continue
            lx0, ly0 = int(x0 * kx), int(y0 * ky)
            lx1, ly1 = min(mw, int(x1 * kx) + 2), min(mh, int(y1 * ky) + 2)
            patch = low8[ly0:ly1, lx0:lx1]
            lo, hi = cv2.minMaxLoc(patch)[:2]
            if hi < 8:  # nobody there: a plain blit
                blit_premult(out, pm, inv, x, y)
                continue
            if lo > 252:  # entirely behind the person: nothing would show
                continue
            roi = out[y0:y1, x0:x1]
            saved = roi.copy()
            blit_premult(out, pm, inv, x, y)
            m = mask_roi(low, x0, y0, x1 - x0, y1 - y0, W, H)
            cv2.blendLinear(saved, roi, m, 1.0 - m, dst=roi)

    def _draw_stamp(self, out: np.ndarray, t: float, stamp: _Stamp | None) -> None:
        if stamp is None:
            return
        H, W = out.shape[:2]
        st = t - STAMP_T0
        if st < 0:
            return
        cx, cy = STAMP_POS[0] * W, STAMP_POS[1] * H
        if st < SLAM_S:
            k = st / SLAM_S
            if stamp.pm.size:
                _blit_scaled(out, stamp.pm, stamp.inv, STAMP_FROM + (1 - STAMP_FROM) * k * k, cx, cy, 0.3 + 0.7 * k)
            return
        after = st - SLAM_S
        if after >= STAMP_HOLD_S + STAMP_FADE_S:
            return
        alpha = 1.0 - ease_in_out_sine((after - STAMP_HOLD_S) / STAMP_FADE_S) if after > STAMP_HOLD_S else 1.0
        dx, dy = shake_offset(after, SHAKE_PX, SHAKE_DECAY)
        sh = 0
        if stamp.pm.size:
            pm, inv = _fade_sprite(stamp.pm, stamp.inv, alpha)
            sh = pm.shape[0]
            blit_premult(out, pm, inv, round(cx - pm.shape[1] / 2) + dx, round(cy - sh / 2) + dy)
        if stamp.caption is not None:
            cpm, cinv = _fade_sprite(*stamp.caption, min(alpha, clamp01(after / CAPTION_FADE_S)))
            ch, cw = cpm.shape[:2]
            top = cy + sh * 0.42 if sh else cy - ch / 2
            blit_premult(out, cpm, cinv, round(cx - cw / 2), round(min(top, H - ch)))


def mask_roi(low: np.ndarray, x0: int, y0: int, w: int, h: int, W: int, H: int) -> np.ndarray:
    """The person mask for one w x h output rectangle, sampled straight from the low-res mask.

    Same bilinear, half-pixel-centred result as upsampling the whole mask to W x H and
    cropping, at the cost of the rectangle only.
    """
    mh, mw = low.shape[:2]
    kx, ky = mw / W, mh / H
    M = np.float32([[kx, 0, (x0 + 0.5) * kx - 0.5], [0, ky, (y0 + 0.5) * ky - 0.5]])
    return cv2.warpAffine(low, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def _reach(depth: np.ndarray, height: int) -> np.ndarray:
    """Conservative half-extent of any pose at this depth (a bill tilted 50 degrees), for culling."""
    return np.where(depth == 1, BILL_W[1], BILL_W[0]) * height * 0.56
