"""Snap once and you crumble to dust; snap again and you come back.

How it stays cheap at 1280x720:

* The room behind you comes from the BackgroundPlate. While you are gone the
  LIVE person mask keeps choosing plate pixels wherever you actually are, so
  you can move and stay invisible while the rest of the room stays live.
* Dissolve freezes the frame and its soft person mask and gives every pixel a
  burn time D: distance from the snap point plus two octaves of noise. The
  frozen you is drawn where D > t with a thin ember band at the burning edge;
  burnt pixels become ash flakes (one per 2x2 cell) that drift away. Flakes are
  sorted by burn time so the live ones are one contiguous slice, and drawing
  them is a gather from a colour table plus np.put into the frame through a
  3-byte void view (6-byte pixel pairs for the young 2x2 flakes). Everything
  else is cv2 on the person's bounding box, with no per-frame float maps.
* Reassembling is the same particle system run backwards from a fresh
  snapshot, then a 150 ms crossfade to live. Toggling mid-way reverses the
  running system in place, so the picture never jumps.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from zap.effects.base import Effect, FxContext, Layer
from zap.effects.plate import BackgroundPlate
from zap.effects.registry import register

log = logging.getLogger(__name__)

VISIBLE, DISSOLVING, VANISHED, REASSEMBLING = "visible", "dissolving", "vanished", "reassembling"

# burn-time field D = T0 + TS*s + TN*(0.75*n1 + 0.25*n2)
T0, TS, TN = 0.10, 1.40, 0.55
Y_WEIGHT = 0.6  # the burn front travels faster vertically
EMBER_BAND_S = 0.035  # a thin charred edge, not an orange band (the film has no orange)
CHAR_BGR = (22, 30, 46)  # the breaking edge darkens to char
ASH_BGR = (34, 36, 40)
K = 8  # colour steps per flake lifetime
K_BIG = K - 2  # flakes stay 2x2 for all but their last steps
LIFE = (1.2, 2.3)
# wind: flakes are carried away from the body, sideways and up, accelerating
WIND_V, WIND_V_SD = 150.0, 60.0  # px/s sideways
WIND_ACC = 130.0  # px/s^2 sideways
RISE_V, RISE_V_SD = -95.0, 40.0  # px/s (up is negative)
RISE_ACC = -45.0  # px/s^2
REASSEMBLE_S = 2.0
CROSSFADE_S = 0.15
PAD_PX = 30  # room for the widened body alpha below
EDGE_GROW_PX = 28  # the frozen body covers the same widened area the hide mask removes: no halo
MAX_PARTICLES = 90_000
A_CORE = 0.5  # soft mask level that counts as "person" for the bbox and the flakes
HIDE_THRESHOLD = 0.25
HIDE_DILATE_PX = 9  # at a 640-wide mask (~18 px at 720p); the hide mask is built at half that resolution
HIDE_BLUR = 5  # at half mask resolution
EMA_ALPHA = 0.5


@dataclass
class _Dust:
    """One frozen snapshot turned into a burn field and a flake system."""

    x0: int
    y0: int
    x1: int
    y1: int
    frozen: np.ndarray  # uint8 ROI of the snapshot
    ember: np.ndarray  # uint8 ROI, snapshot tinted orange
    A: np.ndarray  # float32 ROI soft alpha
    inv_a: np.ndarray  # 1 - A
    D: np.ndarray  # float32 ROI burn time
    d_max: float  # intact region is gone after this
    t_end: float  # everything (flakes included) is gone after this
    col_max: np.ndarray  # per ROI column / row: latest burn time of a visible pixel (shrinks the intact pass)
    row_max: np.ndarray
    # flakes, sorted by burn time d
    px: np.ndarray
    py: np.ndarray
    d: np.ndarray
    kl: np.ndarray  # K / lifetime
    life: np.ndarray
    vx: np.ndarray
    hx: np.ndarray  # half the x acceleration
    vy: np.ndarray
    w: np.ndarray  # swirl angular speed
    ph: np.ndarray
    amp: np.ndarray
    tab: np.ndarray  # (N*K,) V3 colour table, index i*K + k (a flake's steps are adjacent in memory)
    tab2: np.ndarray  # (N*K/2,) V6 table of the young steps, each colour twice (a horizontal pixel pair)
    life_max: float
    # flat per-frame scratch, viewed at the size of the still-intact sub-ROI
    keep: np.ndarray  # uint8 D > t
    band: np.ndarray  # uint8 ember band
    src: np.ndarray  # frozen with the ember band painted in
    tmp: np.ndarray  # src blended over the frame

    @property
    def n(self) -> int:
        return int(self.d.size)


def _squeeze(mask: np.ndarray) -> np.ndarray:
    return mask[..., 0] if mask.ndim == 3 else mask


def _roi_upsample(low: np.ndarray, box: tuple[int, int, int, int], W: int, H: int,
                  buf: np.ndarray | None = None):
    """Crop a low-res map to box (lx0, ly0, lx1, ly1) and resize it to the matching full-res ROI.

    With `buf` (a flat array of at least W*H) the result is a contiguous view into it, so the
    per-frame path allocates nothing big.
    """
    lx0, ly0, lx1, ly1 = box
    sx, sy = W / low.shape[1], H / low.shape[0]
    x0, y0 = round(lx0 * sx), round(ly0 * sy)
    x1, y1 = min(W, round(lx1 * sx)), min(H, round(ly1 * sy))
    dst = None if buf is None else buf[:(y1 - y0) * (x1 - x0)].reshape(y1 - y0, x1 - x0)
    roi = cv2.resize(low[ly0:ly1, lx0:lx1], (x1 - x0, y1 - y0), dst=dst, interpolation=cv2.INTER_LINEAR)
    return (x0, y0, x1, y1), roi


def _scratch(buf: np.ndarray, h: int, w: int, *rest: int) -> np.ndarray:
    """Contiguous (h, w, *rest) view at the start of a flat scratch buffer."""
    n = h * w * math.prod(rest)
    return buf[:n].reshape(h, w, *rest)


@register("presence")
class Presence(Effect):
    """The always-installed presence layer: visible -> dissolving -> vanished -> reassembling -> visible."""

    layer = Layer.PRESENCE

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.active = True  # always installed; never deactivates
        self.state = VISIBLE
        self.stride = int(params.get("stride", 2))
        self.max_particles = int(params.get("max_particles", MAX_PARTICLES))
        self._rng = np.random.default_rng(params.get("seed"))
        self._dust: _Dust | None = None
        self._anchor_now = 0.0
        self._anchor_t = 0.0
        self._rate = 1.0
        self._ema: np.ndarray | None = None
        self._hide: tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None = None
        self._hide_obj: np.ndarray | None = None
        self._hide_seq = -1
        self._mbuf: np.ndarray | None = None  # flat float32 scratch for the hide mask and its inverse
        self._ibuf: np.ndarray | None = None
        self._own_plate: BackgroundPlate | None = None
        self._resized_plate: np.ndarray | None = None
        self.setup_ms = 0.0  # cost of the last particle build, for the perf HUD

    # ------------------------------------------------------------ API
    @property
    def needs_mask(self) -> bool:  # type: ignore[override]
        return self.state != VISIBLE

    @property
    def hidden(self) -> bool:
        """True once the user is (at least partly) replaced by the room."""
        return self.state != VISIBLE

    def set_stride(self, stride: int) -> None:
        """Flake grid stride for the degrade ladder: 2 (default) or 3 (about half the flakes)."""
        if stride not in (2, 3):
            raise ValueError("stride must be 2 or 3")
        self.stride = stride

    def start(self, ctx: FxContext, **kw: Any) -> None:
        """Presence is installed, not started; starting it toggles."""
        self.toggle(ctx, kw.get("where"))

    def toggle(self, ctx: FxContext, where: tuple[float, float] | None = None) -> bool:
        if self.state in (VISIBLE, REASSEMBLING):
            return self.vanish(ctx, where)
        return self.reappear(ctx, where)

    def vanish(self, ctx: FxContext, where: tuple[float, float] | None = None) -> bool:
        """Start dissolving (or reverse a reassembly). Returns False when nothing happened."""
        if self.state in (DISSOLVING, VANISHED):
            return False
        if self.state == REASSEMBLING and self._dust is not None:
            self._set_clock(ctx.now, max(0.0, self._t(ctx.now)), 1.0)
        else:
            dust = self._build(ctx, where)
            if dust is None:
                return False
            self._dust = dust
            self._set_clock(ctx.now, 0.0, 1.0)
        self.state = DISSOLVING
        ctx.sfx.play("presence")
        return True

    def reappear(self, ctx: FxContext, where: tuple[float, float] | None = None) -> bool:
        """Start reassembling (or reverse a dissolve). Returns False when nothing happened."""
        if self.state in (VISIBLE, REASSEMBLING):
            return False
        if self.state == DISSOLVING and self._dust is not None:
            dust = self._dust
            self._set_clock(ctx.now, min(self._t(ctx.now), dust.t_end), -dust.t_end / REASSEMBLE_S)
        else:
            dust = self._build(ctx, where)
            if dust is None:
                # nobody to reassemble (no mask, or out of frame): live already shows the truth
                self.reset()
                ctx.sfx.play("reappear")
                return True
            self._dust = dust
            self._set_clock(ctx.now, dust.t_end, -dust.t_end / REASSEMBLE_S)
        self.state = REASSEMBLING
        ctx.sfx.play("reappear")
        return True

    def reset(self) -> None:
        """Instantly visible (PANIC, crash recovery)."""
        self.state = VISIBLE
        self._dust = None

    def start_hidden(self) -> None:
        """Instantly vanished, without the dissolve (snap entry): the meeting sees only the room
        until ``reappear()``. Needs the person mask every frame, like any hidden state."""
        self.state = VANISHED
        self._dust = None

    def abort(self) -> None:
        self.reset()  # presence stays installed and active

    # ------------------------------------------------------------ frame
    def update(self, ctx: FxContext, out: np.ndarray) -> None:
        self._track_mask(ctx)
        if self.state == VISIBLE:
            return
        if self._own_plate is not None and ctx.plate is None and ctx.sig.mask is not None and ctx.sig.mask_fresh:
            self._own_plate.update(ctx.frame, ctx.sig.mask, ctx.now)
        if not out.flags.c_contiguous:
            tmp = np.ascontiguousarray(out)
            self._draw(ctx, tmp)
            out[...] = tmp
        else:
            self._draw(ctx, out)

    def _draw(self, ctx: FxContext, out: np.ndarray) -> None:
        if self.state == VANISHED:
            self._hide_live(ctx, out)
            return
        dust = self._dust
        if dust is None:
            self.reset()
            return
        t = self._t(ctx.now)
        if self.state == DISSOLVING:
            if t >= dust.t_end:
                self.state = VANISHED
                self._dust = None
                self._hide_live(ctx, out)
                return
            self._render(ctx, out, dust, t)
            return
        # reassembling
        if t > 0.0:
            self._render(ctx, out, dust, t)
            return
        t_zero = self._anchor_now + self._anchor_t / -self._rate
        f = (ctx.now - t_zero) / CROSSFADE_S
        if f >= 1.0:
            self.reset()
            return
        self._render(ctx, out, dust, 0.0)
        x0, y0, x1, y1 = self._union_box(dust)
        o = out[y0:y1, x0:x1]
        cv2.addWeighted(o, 1.0 - max(0.0, f), ctx.frame[y0:y1, x0:x1], max(0.0, f), 0.0, dst=o)

    def _render(self, ctx: FxContext, out: np.ndarray, dust: _Dust, t: float) -> None:
        self._hide_live(ctx, out)
        if t < dust.d_max:
            self._draw_intact(out, dust, t)
        self._draw_flakes(out, dust, t)

    # ------------------------------------------------------------ clock
    def _set_clock(self, now: float, t: float, rate: float) -> None:
        self._anchor_now, self._anchor_t, self._rate = now, t, rate

    def _t(self, now: float) -> float:
        return self._anchor_t + self._rate * (now - self._anchor_now)

    # ------------------------------------------------------------ masks
    def _track_mask(self, ctx: FxContext) -> None:
        """EMA of the low-res person mask, so the frozen silhouette is not one noisy frame."""
        m = ctx.sig.mask
        if m is None or not ctx.sig.mask_fresh:
            return
        m = _squeeze(m)
        if self._ema is None or self._ema.shape != m.shape:
            self._ema = m.astype(np.float32, copy=True)
        else:
            cv2.accumulateWeighted(m, self._ema, EMA_ALPHA)

    def _hide_mask(self, ctx: FxContext):
        """(box, M, 1-M) at output resolution from the live mask, dilated and softened; cached per mask."""
        mask = ctx.sig.mask
        if mask is None:
            return self._hide
        if mask is self._hide_obj and not (ctx.sig.mask_fresh and ctx.sig.seq != self._hide_seq):
            return self._hide
        self._hide_obj, self._hide_seq = mask, ctx.sig.seq
        m = _squeeze(mask)
        half = cv2.resize(m, (max(1, m.shape[1] // 2), max(1, m.shape[0] // 2)), interpolation=cv2.INTER_AREA)
        hard = cv2.compare(half, HIDE_THRESHOLD, cv2.CMP_GT)
        r = max(1, int(HIDE_DILATE_PX * half.shape[1] / 640 + 0.5))
        hard = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        soft = cv2.GaussianBlur(hard, (HIDE_BLUR, HIDE_BLUR), 0)
        bx, by, bw, bh = cv2.boundingRect(soft)
        if bw == 0 or bh == 0:
            self._hide = None
            return None
        n = ctx.W * ctx.H
        if self._mbuf is None or self._mbuf.size < n:
            self._mbuf, self._ibuf = np.empty(n, np.float32), np.empty(n, np.float32)
        low = soft.astype(np.float32) * np.float32(1 / 255)  # quarter-size map: cheap
        box, M = _roi_upsample(low, (bx, by, bx + bw, by + bh), ctx.W, ctx.H, self._mbuf)
        inv = _scratch(self._ibuf, *M.shape)
        np.subtract(np.float32(1.0), M, out=inv)
        self._hide = (box, M, inv)
        return self._hide

    def _plate_image(self, ctx: FxContext) -> np.ndarray:
        plate = ctx.plate if ctx.plate is not None else self._own_plate
        if plate is None:
            log.warning("presence: no background plate supplied; using a private one")
            plate = self._own_plate = BackgroundPlate(ctx.W, ctx.H)
            plate.update(ctx.frame, ctx.sig.mask, ctx.now)
        img = plate.image()
        if img.shape != ctx.frame.shape:
            if self._resized_plate is None or self._resized_plate.shape != ctx.frame.shape:
                log.warning("presence: plate is %s but frames are %s; resizing", img.shape, ctx.frame.shape)
            self._resized_plate = cv2.resize(img, (ctx.W, ctx.H), interpolation=cv2.INTER_LINEAR)
            img = self._resized_plate
        return img

    def _hide_live(self, ctx: FxContext, out: np.ndarray) -> None:
        """out = plate*M + live*(1-M) where the live person is, so they stay invisible."""
        plate = self._plate_image(ctx)
        hide = self._hide_mask(ctx)
        if hide is None:
            if ctx.sig.mask is None:
                np.copyto(out, plate)  # never had a mask: hide everything rather than show the user
            return
        (x0, y0, x1, y1), M, inv = hide
        o = out[y0:y1, x0:x1]
        cv2.blendLinear(plate[y0:y1, x0:x1], o, M, inv, dst=o)

    def _union_box(self, dust: _Dust) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = dust.x0, dust.y0, dust.x1, dust.y1
        if self._hide is not None:
            hx0, hy0, hx1, hy1 = self._hide[0]
            x0, y0, x1, y1 = min(x0, hx0), min(y0, hy0), max(x1, hx1), max(y1, hy1)
        return x0, y0, x1, y1

    # ------------------------------------------------------------ build
    def _build(self, ctx: FxContext, where: tuple[float, float] | None) -> _Dust | None:
        """Freeze the current frame and precompute the burn field and the flakes."""
        t_start = time.perf_counter()
        mask = ctx.sig.mask
        if mask is None:
            log.warning("presence: no person mask available; staying as is")
            return None
        m = _squeeze(mask)
        low = np.maximum(self._ema, m) if self._ema is not None and self._ema.shape == m.shape else m
        low = cv2.GaussianBlur(np.ascontiguousarray(low, dtype=np.float32), (5, 5), 0)
        bx, by, bw, bh = cv2.boundingRect(cv2.compare(low, A_CORE, cv2.CMP_GT))
        if bw == 0 or bh == 0:
            log.info("presence: no person in the mask; nothing to dissolve")
            return None
        frame = np.ascontiguousarray(ctx.frame)
        H, W = frame.shape[:2]
        mh, mw = low.shape
        px_pad = math.ceil(PAD_PX * mw / W) + 2  # 8 output px plus the blur falloff
        py_pad = math.ceil(PAD_PX * mh / H) + 2
        lbox = (max(0, bx - px_pad), max(0, by - py_pad), min(mw, bx + bw + px_pad), min(mh, by + bh + py_pad))
        (x0, y0, x1, y1), A = _roi_upsample(low, lbox, W, H)
        h, w = A.shape
        grow = max(3, round(EDGE_GROW_PX * W / 1280))
        A_draw = cv2.GaussianBlur(cv2.dilate(A, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)),
                                  (0, 0), grow / 4)
        np.maximum(A_draw, A, out=A_draw)
        if where is None:
            ix, iy = (bx + bw) * W / mw, (by + bh / 2) * H / mh  # right-middle of the person
        else:
            ix, iy = min(max(float(where[0]), 0.0), W - 1.0), min(max(float(where[1]), 0.0), H - 1.0)

        # burn time field over the ROI
        dx = (np.arange(x0, x1, dtype=np.float32) - np.float32(ix)) ** 2
        dy = (np.float32(Y_WEIGHT) * (np.arange(y0, y1, dtype=np.float32) - np.float32(iy))) ** 2
        dist = cv2.sqrt(dy[:, None] + dx[None, :])
        core = cv2.compare(A, A_CORE, cv2.CMP_GT)
        smax = max(1.0, cv2.minMaxLoc(dist, mask=core)[1])
        rng = self._rng
        n1 = cv2.resize(rng.random((6, 8), dtype=np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        np.clip(n1, 0.0, 1.0, out=n1)
        n2 = cv2.resize(rng.random((max(2, h // 6), max(2, w // 6)), dtype=np.float32), (w, h),
                        interpolation=cv2.INTER_LINEAR)
        D = cv2.addWeighted(n1, 0.75 * TN, n2, 0.25 * TN, T0)
        cv2.scaleAdd(dist, TS / smax, D, dst=D)
        visible = cv2.compare(A_draw, 1 / 255, cv2.CMP_GT)
        d_max = float(cv2.minMaxLoc(D, mask=visible)[1]) if cv2.countNonZero(visible) else 0.0
        Dv = np.where(visible > 0, D, np.float32(-1.0))

        frozen = frame[y0:y1, x0:x1].copy()
        char = np.empty_like(frozen)
        char[:] = CHAR_BGR
        ember = cv2.addWeighted(frozen, 0.35, char, 1.0, 0.0)

        flakes = self._flakes(frame, A, D, x0, y0, ix)
        life_max = float(flakes["life"].max()) if flakes["life"].size else 0.0
        t_end = max(d_max, float((flakes["d"] + flakes["life"]).max()) if flakes["d"].size else 0.0) + 1e-3
        dust = _Dust(
            x0=x0, y0=y0, x1=x1, y1=y1, frozen=frozen, ember=ember, A=A_draw,
            inv_a=np.subtract(np.float32(1.0), A_draw),
            D=D, d_max=d_max, t_end=t_end, col_max=Dv.max(axis=0), row_max=Dv.max(axis=1), life_max=life_max,
            **flakes, keep=np.empty(h * w, np.uint8), band=np.empty(h * w, np.uint8),
            src=np.empty(h * w * 3, np.uint8), tmp=np.empty(h * w * 3, np.uint8),
        )
        self.setup_ms = (time.perf_counter() - t_start) * 1e3
        log.debug("presence: %d flakes, roi %dx%d, %.2f s, setup %.1f ms", dust.n, w, h, t_end, self.setup_ms)
        return dust

    def _flakes(self, frame: np.ndarray, A: np.ndarray, D: np.ndarray, x0: int, y0: int,
                ix: float) -> dict[str, np.ndarray]:
        st, rng = self.stride, self._rng
        gy, gx = np.nonzero(A[::st, ::st] > A_CORE)
        if gy.size > self.max_particles:
            keep = np.sort(rng.choice(gy.size, self.max_particles, replace=False))
            gy, gx = gy[keep], gx[keep]
        ry, rx = gy * st, gx * st
        n = int(ry.size)
        d = D[ry, rx]
        order = np.argsort(d, kind="stable")
        ry, rx, d = ry[order], rx[order], d[order]
        W = frame.shape[1]
        flat = (ry + y0) * W + (rx + x0)
        col = np.take(frame.reshape(-1, 3).view("V3").ravel(), flat).view(np.uint8).reshape(n, 3)

        f32 = np.float32
        px = (rx + x0).astype(f32)
        py = (ry + y0).astype(f32)
        life = rng.uniform(*LIFE, n).astype(f32)
        dirx = np.tanh((px - f32(ix)) / f32(60.0))  # drift away from the ignition side
        vx = (rng.normal(WIND_V, WIND_V_SD, n).astype(f32) * dirx)
        hx = f32(0.5 * WIND_ACC) * dirx
        vy = rng.normal(RISE_V, RISE_V_SD, n).astype(f32)
        w = rng.uniform(3.0, 7.0, n).astype(f32)
        ph = rng.uniform(0.0, 2 * math.pi, n).astype(f32)
        amp = rng.uniform(3.0, 12.0, n).astype(f32)

        # colour table: true colour (briefly hot) -> darker ash over K steps
        colf = col.astype(f32)
        ash = np.array(ASH_BGR, f32)
        tab = np.empty((n, K, 3), np.uint8)
        for k in range(K):
            f = min(1.0, 0.35 + 0.65 * (k / (K - 1)) ** 0.6)  # quickly charred to dark ash
            c = (colf * f32(1 - f) + ash * f32(f)) * f32(1.0 - 0.25 * k / (K - 1))
            np.clip(c, 0, 255, out=c)
            tab[:, k] = c
        tab2 = np.concatenate([tab[:, :K_BIG], tab[:, :K_BIG]], axis=2)  # (n, K_BIG, 6)
        return {
            "px": px, "py": py, "d": d.astype(f32), "kl": (f32(K) / life), "life": life,
            "vx": vx, "hx": hx, "vy": vy, "w": w, "ph": ph, "amp": amp,
            "tab": tab.reshape(-1, 3).view("V3").ravel(), "tab2": tab2.reshape(-1, 6).view("V6").ravel(),
        }

    # ------------------------------------------------------------ drawing
    def _draw_intact(self, out: np.ndarray, dust: _Dust, t: float) -> None:
        """Frozen snapshot where D > t (soft-edged by A) with an orange ember band at the front."""
        cols = np.flatnonzero(dust.col_max > t)  # the still-intact part shrinks as the front advances
        rows = np.flatnonzero(dust.row_max > t)
        if cols.size == 0 or rows.size == 0:
            return
        c0, c1, r0, r1 = int(cols[0]), int(cols[-1]) + 1, int(rows[0]), int(rows[-1]) + 1
        h, w = r1 - r0, c1 - c0
        D = dust.D[r0:r1, c0:c1]
        keep, band = _scratch(dust.keep, h, w), _scratch(dust.band, h, w)
        src, tmp = _scratch(dust.src, h, w, 3), _scratch(dust.tmp, h, w, 3)
        cv2.compare(D, t, cv2.CMP_GT, dst=keep)
        cv2.inRange(D, t, t + EMBER_BAND_S, dst=band)
        np.copyto(src, dust.frozen[r0:r1, c0:c1])
        cv2.copyTo(dust.ember[r0:r1, c0:c1], band, src)
        # blend with the fixed soft alpha, then keep only the unburnt part: no per-frame float maps
        o = out[dust.y0 + r0:dust.y0 + r1, dust.x0 + c0:dust.x0 + c1]
        cv2.blendLinear(src, o, dust.A[r0:r1, c0:c1], dust.inv_a[r0:r1, c0:c1], dst=tmp)
        cv2.copyTo(tmp, keep, o)

    def _draw_flakes(self, out: np.ndarray, dust: _Dust, t: float) -> None:
        if dust.n == 0:
            return
        lo = int(np.searchsorted(dust.d, t - dust.life_max, side="right"))
        hi = int(np.searchsorted(dust.d, t, side="right"))
        if hi <= lo:
            return
        f32 = np.float32
        a = f32(t) - dust.d[lo:hi]
        life = dust.life[lo:hi]
        arg = dust.w[lo:hi] * a
        arg += dust.ph[lo:hi]
        amp_a = dust.amp[lo:hi] * a
        x = dust.hx[lo:hi] * a
        x += dust.vx[lo:hi]
        x *= a
        x += dust.px[lo:hi]
        x += amp_a * np.sin(arg)
        y = f32(0.5 * RISE_ACC) * a  # upward acceleration
        y += dust.vy[lo:hi]
        y *= a
        y += dust.py[lo:hi]
        y += f32(0.5) * amp_a * np.cos(arg)
        H, W = out.shape[:2]
        ok = a < life
        ok &= x >= 0
        ok &= x < W - 1
        ok &= y >= 0
        ok &= y < H - 1
        # colour step and pixel index for the whole slice; only the drawn ones are gathered below
        k = (a * dust.kl[lo:hi]).astype(np.intp)
        np.minimum(k, K - 1, out=k)
        flat = y.astype(np.intp)
        flat *= W
        flat += x.astype(np.intp)
        young = k < K_BIG  # flakes are 2x2 (they survive Zoom's encoder) until their last steps
        if W % 2:  # odd width: no pixel-pair view, draw every flake as one pixel
            young[:] = False
        old = np.flatnonzero(ok & ~young)
        yi = np.flatnonzero(ok & young)
        if old.size:
            g = (old + lo) * K + np.take(k, old)
            ov = out.reshape(-1, 3).view("V3").ravel()
            np.put(ov, np.take(flat, old), np.take(dust.tab, g))  # np.put: ~2x faster than setitem on void
        if yi.size:
            # x snapped to even: each row of the 2x2 block is one 6-byte pixel pair
            g = (yi + lo) * K_BIG + np.take(k, yi)
            cols = np.take(dust.tab2, g)
            pair = np.take(flat, yi) >> 1
            ov2 = out.reshape(-1, 6).view("V6").ravel()
            np.put(ov2, pair, cols)
            pair += W // 2
            np.put(ov2, pair, cols)
