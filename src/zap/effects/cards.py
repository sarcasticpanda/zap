"""Freeze cards: "We'll Be Right Back" (peace sign) and "To Be Continued" (L sign).

Both freeze the picture the meeting was seeing, play their music and go back
to live when it ends; an open palm (or the hotkey again) ends them early.

* We'll Be Right Back, the Eric Andre Show bumper: the camera snaps in on your
  face with a little shake and a chunky yellow caption pops up. If you have
  walked off when the music ends, the card stays up until you are back in view.
* To Be Continued, the JoJo ending: the colour drains to sepia with a small
  thunk and the arrow slides in from the left.

The freeze is taken from ``out`` on the first frame, i.e. exactly what the
meeting was seeing, and graded once, so each frame costs one warp and a sprite
blit. The songs are copyrighted, so the music is the user's own ``brb`` / ``tbc``
file in assets/sfx; without one, a procedural stand-in plays.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from zap.effects.base import Effect, FxContext, Layer
from zap.effects.cutaway import blit_scaled
from zap.effects.registry import register
from zap.render.blend import (
    blit_premult,
    clamp01,
    ease_in_out_sine,
    ease_out_back,
    ease_out_cubic,
    make_premult,
    shake_offset,
    vignette_map,
)
from zap.render.text import TextStyle, load_font, render_text

log = logging.getLogger(__name__)

SHOW, AWAY, OUT = "show", "away", "out"

OUT_S = 0.55  # back to live: the card zooms out over the first OUT_ZOOM of it ...
OUT_ZOOM, OUT_FADE = 0.65, 0.45  # ... and fades to live over the last OUT_FADE (shares of OUT_S)
AWAY_S = 1.0  # no face this long when the music ends = you left: hold the card (if the card does that)
BACK_S = 0.8  # your face seen this long = you're back
MIN_S, MAX_S = 2.0, 12.0
FLASH_S = 0.08

# We'll Be Right Back
PUNCH_S = 0.16  # the snap zoom onto your face
FACE_W = 0.30  # after the punch your face is this wide (fraction of the frame width) ...
ZOOM_MIN, ZOOM_MAX = 1.3, 2.4  # ... within these zooms
FACE_AT = (0.5, 0.40)  # ... and centred here, above the caption
CREEP, CREEP_S = 0.08, 6.0  # then a slow push-in by this much over this long
SHAKE_S, SHAKE_PX = 0.45, 9.0
CAPTION_AT, CAPTION_POP_S = 0.1, 0.3
CAPTION_Y = 0.80  # caption centre, fraction of the frame height
CAPTION_FONT = "ariblk.ttf"  # Arial Black: chunky, and on every Windows PC
CAPTION_FILL = (255, 212, 0, 255)  # RGBA yellow
CAPTION_TILT = 4.0  # degrees, rising to the right

# To Be Continued. The standard sepia matrix rearranged for BGR in and out.
SEPIA_BGR = np.float32([
    [0.131, 0.534, 0.272],
    [0.168, 0.686, 0.349],
    [0.189, 0.769, 0.393],
])
SEPIA_S = 0.4  # colour -> sepia
THUNK_S = 0.14  # the freeze "thunk": a small punch and flash
ARROW_IN_S = 0.6
ARROW_FILL = (176, 158, 88, 255)  # olive/khaki, RGBA
ARROW_EDGE = (66, 46, 20, 255)  # dark brown
ARROW_FONT = "georgiaz.ttf"  # Georgia Bold Italic


class FreezeCard(Effect):
    """Freeze, draw the card, music; ``seconds = 0`` lasts as long as the music (``sfx_key``)."""

    layer = Layer.CUTAWAY
    needs_face = True  # where your face is, and whether you are still there
    default_s = 4.0  # when the music's length is unknown
    min_s = MIN_S  # a short sound still shows the card this long

    def __init__(self, seconds: float = 0.0, hold_when_away: bool = False, **params: Any) -> None:
        super().__init__(seconds=seconds, hold_when_away=hold_when_away, **params)
        self.seconds = float(seconds)
        self.hold_when_away = bool(hold_when_away)
        self.state = SHOW
        self.duration = self.default_s
        self._freeze: np.ndarray | None = None
        self._buf: np.ndarray | None = None
        self._face_t = -math.inf
        self._back_since: float | None = None
        self._t_out = 0.0
        self._z_out = 1.0

    # ------------------------------------------------------------ lifecycle
    def start(self, ctx: FxContext, **kw: Any) -> None:
        super().start(ctx, **kw)
        self.state = SHOW
        self._freeze = None
        self._back_since = None
        self.duration = self._length(ctx)
        self._face_t = ctx.now if ctx.sig.face is not None and ctx.sig.face.fresh else -math.inf

    def _length(self, ctx: FxContext) -> float:
        if self.seconds > 0:
            return min(max(self.seconds, self.min_s), MAX_S)
        length = getattr(ctx.sfx, "length", None)
        music = float(length(self.sfx_key)) if length is not None and self.sfx_key else 0.0
        return min(max(music, self.min_s), MAX_S) if music > 0 else self.default_s

    def finish(self, ctx: FxContext) -> None:
        """Back to live now (an open palm, or the hotkey again)."""
        if self.active and self.state != OUT:
            self._z_out = self._zoom_at(ctx.now - self.t0)
            self.state, self._t_out = OUT, ctx.now

    def update(self, ctx: FxContext, out: np.ndarray) -> None:
        now = ctx.now
        t = now - self.t0
        if self._freeze is None:
            self._capture(out)
            self._buf = np.empty_like(out)
        face = ctx.sig.face
        if face is not None and face.fresh:
            self._face_t = now
            self._back_since = self._back_since if self._back_since is not None else now
        elif now - self._face_t > 0.3:  # the face model drops a frame now and then
            self._back_since = None

        if self.state == SHOW and t >= self.duration:
            if self.hold_when_away and now - self._face_t > AWAY_S:
                self.state = AWAY
                log.info("%s: you left; holding the card until you're back", self.name)
            else:
                self.finish(ctx)
        if self.state == AWAY and self._back_since is not None and now - self._back_since >= BACK_S:
            log.info("%s: you're back", self.name)
            self.finish(ctx)

        if self.state == OUT:
            k = (now - self._t_out) / OUT_S
            if k >= 1.0:
                self.active = False
                return  # `out` already holds the live picture
            # zoom the card back out first, then fade: a fade while zoomed in shows two faces
            e = ease_in_out_sine(min(1.0, k / OUT_ZOOM))
            self._draw(self._buf, t, 1.0 + (self._z_out - 1.0) * (1.0 - e), overlay=1.0 - e)
            fade = clamp01((k - (1.0 - OUT_FADE)) / OUT_FADE)
            cv2.addWeighted(self._buf, 1.0 - fade, out, fade, 0, dst=out)
            return
        self._draw(out, t, self._zoom_at(t), overlay=1.0)

    # ------------------------------------------------------------ per card
    def _capture(self, out: np.ndarray) -> None:
        self._freeze = out.copy()

    def _zoom_at(self, t: float) -> float:
        return 1.0

    def _draw(self, dst: np.ndarray, t: float, z: float, overlay: float) -> None:
        raise NotImplementedError


@register("brb")
class BeRightBack(FreezeCard):
    """Freeze, punch in on your face, yellow caption, bumper music; held while you're away."""

    sfx_key = "brb"
    default_s = 4.5

    def __init__(self, text: str = "We'll Be Right Back", seconds: float = 0.0, hold_when_away: bool = True,
                 **params: Any) -> None:
        super().__init__(seconds=seconds, hold_when_away=hold_when_away, text=text, **params)
        self.text = text
        self._zoom, self._face_c = 1.0, (0.0, 0.0)

    @classmethod
    def prewarm(cls, width: int, height: int, text: str = "We'll Be Right Back") -> None:
        _caption_sprite(text, width)
        _grade_vignette(width, height)

    def start(self, ctx: FxContext, **kw: Any) -> None:
        super().start(ctx, **kw)
        face = ctx.sig.face if ctx.sig.face is not None else ctx.last_face
        W, H = ctx.W, ctx.H
        if face is not None:
            self._zoom = min(max(FACE_W * W / max(face.width, 1.0), ZOOM_MIN), ZOOM_MAX)
            self._face_c = (float(face.center[0]), float(face.center[1]))
        else:
            self._zoom, self._face_c = ZOOM_MIN, (W / 2, H * FACE_AT[1])

    def _capture(self, out: np.ndarray) -> None:
        """Freeze what the meeting sees, graded once: a little punchier, darker at the edges."""
        H, W = out.shape[:2]
        gray = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        punchy = cv2.addWeighted(out, 1.1, gray, -0.1, 0)  # a touch more colour (more turns skin red)
        punchy = cv2.convertScaleAbs(punchy, alpha=1.06, beta=-6)  # a bit more contrast
        self._freeze = cv2.multiply(punchy, _grade_vignette(W, H), scale=1 / 255)

    def _zoom_at(self, t: float) -> float:
        if t < PUNCH_S:
            return 1.0 + (self._zoom - 1.0) * ease_out_cubic(t / PUNCH_S)
        return self._zoom * (1.0 + CREEP * ease_out_cubic(clamp01((t - PUNCH_S) / CREEP_S)))

    def _draw(self, dst: np.ndarray, t: float, z: float, overlay: float) -> None:
        """The frozen frame zoomed by z onto your face, with the shake, flash and caption at time t."""
        H, W = dst.shape[:2]
        k = clamp01(t / PUNCH_S) if self.state == SHOW else 1.0
        cx, cy = self._face_c
        px = cx + (FACE_AT[0] * W - cx) * k
        py = cy + (FACE_AT[1] * H - cy) * k
        # x' = z * x + d puts the face at (px, py); keep the zoomed picture covering the whole frame
        dx = min(max(px - z * cx, W - z * W), 0.0)
        dy = min(max(py - z * cy, H - z * H), 0.0)
        if PUNCH_S <= t < PUNCH_S + SHAKE_S and self.state == SHOW:
            sx, sy = shake_offset(t - PUNCH_S, SHAKE_PX * W / 1280, decay=7.0)
            dx, dy = dx + sx, dy + sy
        M = np.float32([[z, 0, dx], [0, z, dy]])
        cv2.warpAffine(self._freeze, M, (W, H), dst=dst, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if t < FLASH_S:
            f = 90 * (1.0 - t / FLASH_S)
            cv2.add(dst, (f, f, f, 0), dst=dst)
        tc = t - CAPTION_AT
        if tc > 0 and overlay > 0:
            pm, inv3 = _caption_sprite(self.text, W)
            scale = 0.3 + 0.7 * ease_out_back(tc / CAPTION_POP_S, overshoot=2.0)
            blit_scaled(dst, pm, inv3, W / 2, H * CAPTION_Y, scale, min(1.0, tc / 0.06) * overlay)


@register("tbc")
class ToBeContinued(FreezeCard):
    """Freeze, colour drains to sepia with a small thunk, the arrow slides in; music."""

    sfx_key = "tbc"
    min_s = 3.0  # time to see the arrow land

    @classmethod
    def prewarm(cls, width: int, height: int) -> None:
        _sepia_vignette(width, height)
        _arrow_sprite(width)

    def _capture(self, out: np.ndarray) -> None:
        H, W = out.shape[:2]
        self._freeze = out.copy()
        sepia = cv2.transform(self._freeze, SEPIA_BGR)
        self._sepia = cv2.multiply(sepia, _sepia_vignette(W, H), scale=1 / 255)
        self._mix = np.empty_like(out)

    def _zoom_at(self, t: float) -> float:
        return 1.0 + 0.05 * (1.0 - ease_out_cubic(t / THUNK_S)) if t < THUNK_S else 1.0

    def _draw(self, dst: np.ndarray, t: float, z: float, overlay: float) -> None:
        H, W = dst.shape[:2]
        src = self._sepia
        if t < SEPIA_S and self.state == SHOW:
            k = ease_out_cubic(t / SEPIA_S)
            cv2.addWeighted(self._freeze, 1 - k, self._sepia, k, 0, dst=self._mix)
            src = self._mix
        if z > 1.001:
            M = np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H / 2]])
            cv2.warpAffine(src, M, (W, H), dst=dst, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            np.copyto(dst, src)
        if t < FLASH_S:
            f = 70 * (1 - t / FLASH_S)
            cv2.add(dst, (f, f, f, 0), dst=dst)
        if overlay <= 0:
            return
        k = W / 1280
        pm, inv3 = _arrow_sprite(W)
        x_end, x_start = int(36 * k), -pm.shape[1] - 10
        x = int(x_start + (x_end - x_start) * ease_out_back(t / ARROW_IN_S))
        if overlay < 0.996:
            pm = cv2.convertScaleAbs(pm, alpha=overlay)
            inv3 = cv2.convertScaleAbs(inv3, alpha=overlay, beta=255 * (1 - overlay))
        blit_premult(dst, pm, inv3, x, H - pm.shape[0] - int(40 * k))


# ---------------------------------------------------------------- sprites
@lru_cache(maxsize=2)
def _grade_vignette(W: int, H: int) -> np.ndarray:
    return vignette_map(W, H, strength=0.35, radius=0.95)


@lru_cache(maxsize=2)
def _sepia_vignette(W: int, H: int) -> np.ndarray:
    return vignette_map(W, H, strength=0.45, radius=0.9)


@lru_cache(maxsize=8)
def _caption_sprite(text: str, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Chunky yellow letters, a black outline and a hard drop shadow, tilted up a little."""
    k = W / 1280
    style = TextStyle(font=CAPTION_FONT, size=max(16, int(86 * k)), color=CAPTION_FILL,
                      stroke=max(2, int(7 * k)), stroke_color=(0, 0, 0, 255))
    fill = render_text(text, style)
    shadow = fill.copy()
    shadow[..., :3] = 0
    shadow[..., 3] = (shadow[..., 3].astype(np.float32) * 0.85).astype(np.uint8)
    off = max(2, int(7 * k))
    h, w = fill.shape[:2]
    canvas = Image.new("RGBA", (w + off, h + off), (0, 0, 0, 0))
    canvas.alpha_composite(Image.fromarray(shadow), (off, off))
    canvas.alpha_composite(Image.fromarray(fill), (0, 0))
    canvas = canvas.rotate(CAPTION_TILT, resample=Image.BICUBIC, expand=True)
    rgba = np.asarray(canvas).copy()
    max_w = int(0.88 * W)
    if rgba.shape[1] > max_w:
        s = max_w / rgba.shape[1]
        rgba = cv2.resize(rgba, (max_w, max(1, round(rgba.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return make_premult(rgba)


@lru_cache(maxsize=4)
def _arrow_sprite(W: int) -> tuple[np.ndarray, np.ndarray]:
    """The JoJo "To Be Continued" arrow, pointing left, drawn supersampled for clean edges."""
    k, ss = W / 1280, 3
    w, h = max(40, int(520 * k)), max(10, int(104 * k))
    w2, h2 = w * ss, h * ss
    img = Image.new("RGBA", (w2, h2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lw = max(2, int(5 * k)) * ss
    head = int(0.19 * w2)
    sy0, sy1 = int(h2 * 0.2), int(h2 * 0.8)
    pts = [(lw, h2 / 2), (head, lw), (head, sy0), (w2 - lw, sy0), (w2 - lw, sy1), (head, sy1), (head, h2 - lw)]
    d.polygon(pts, fill=ARROW_FILL, outline=ARROW_EDGE, width=lw)
    font = load_font(ARROW_FONT, max(8, int(44 * k * ss)))
    d.text(((head + w2) / 2, h2 / 2), "To Be Continued", font=font, fill=ARROW_EDGE, anchor="mm")
    rgba = np.asarray(img.resize((w, h), Image.LANCZOS)).copy()
    return make_premult(rgba)
