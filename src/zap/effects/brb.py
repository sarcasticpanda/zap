"""'We'll Be Right Back': freeze, punch in on your face, caption and bumper music.

The Eric Andre Show bumper, as the meme uses it: the picture the meeting was
seeing freezes, the camera snaps in on your face with a little shake, the
chunky yellow caption pops up and the music plays. When the music is over and
you are in view, it zooms back out to live. If you have walked off (no face
for a moment), the frozen card stays up as a real "be right back" screen until
you are back in front of the camera.

The freeze is taken from ``out`` on the first frame, i.e. exactly what the
meeting was seeing, and graded once, so each frame costs one warp and one
sprite blit. The song is copyrighted, so the music is the user's own ``brb``
file in assets/sfx; without one, a procedural stand-in plays.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from PIL import Image

from zap.effects.base import Effect, FxContext, Layer
from zap.effects.cutaway import blit_scaled
from zap.effects.registry import register
from zap.render.blend import (
    clamp01,
    ease_in_out_sine,
    ease_out_back,
    ease_out_cubic,
    make_premult,
    shake_offset,
    vignette_map,
)
from zap.render.text import TextStyle, render_text

log = logging.getLogger(__name__)

SHOW, AWAY, OUT = "show", "away", "out"

PUNCH_S = 0.16  # the snap zoom onto your face
FACE_W = 0.30  # after the punch your face is this wide (fraction of the frame width) ...
ZOOM_MIN, ZOOM_MAX = 1.3, 2.4  # ... within these zooms
FACE_AT = (0.5, 0.40)  # ... and centred here, above the caption
CREEP, CREEP_S = 0.08, 6.0  # then a slow push-in by this much over this long
SHAKE_S, SHAKE_PX = 0.45, 9.0
FLASH_S = 0.08
CAPTION_AT, CAPTION_POP_S = 0.1, 0.3
CAPTION_Y = 0.80  # caption centre, fraction of the frame height
CAPTION_FONT = "ariblk.ttf"  # Arial Black: chunky, and on every Windows PC
CAPTION_FILL = (255, 212, 0, 255)  # RGBA yellow
CAPTION_TILT = 4.0  # degrees, rising to the right
OUT_S = 0.55  # back to live: the card zooms out over the first OUT_ZOOM of it ...
OUT_ZOOM, OUT_FADE = 0.65, 0.45  # ... and fades to live over the last OUT_FADE (shares of OUT_S)
AWAY_S = 1.0  # no face this long when the music ends = you left: hold the card
BACK_S = 0.8  # your face seen this long = you're back
DEFAULT_S = 4.5  # when the music's length is unknown
MIN_S, MAX_S = 2.0, 12.0


@register("brb")
class BeRightBack(Effect):
    """Freeze, punch in, caption, music; ``seconds = 0`` lasts as long as the music."""

    layer = Layer.CUTAWAY
    needs_face = True  # where to punch in, and whether you are still there
    sfx_key = "brb"

    def __init__(self, text: str = "We'll Be Right Back", seconds: float = 0.0, hold_when_away: bool = True,
                 **params: Any) -> None:
        super().__init__(text=text, seconds=seconds, hold_when_away=hold_when_away, **params)
        self.text = text
        self.seconds = float(seconds)
        self.hold_when_away = bool(hold_when_away)
        self.state = SHOW
        self.duration = DEFAULT_S
        self._freeze: np.ndarray | None = None
        self._buf: np.ndarray | None = None
        self._zoom, self._face_c = 1.0, (0.0, 0.0)
        self._face_t = -math.inf
        self._back_since: float | None = None
        self._t_out = 0.0
        self._z_out = 1.0

    @classmethod
    def prewarm(cls, width: int, height: int, text: str = "We'll Be Right Back") -> None:
        _caption_sprite(text, width)
        _grade_vignette(width, height)

    # ------------------------------------------------------------ lifecycle
    def start(self, ctx: FxContext, **kw: Any) -> None:
        super().start(ctx, **kw)
        self.state = SHOW
        self._freeze = None
        self._back_since = None
        self.duration = self._length(ctx)
        face = ctx.sig.face if ctx.sig.face is not None else ctx.last_face
        self._face_t = ctx.now if ctx.sig.face is not None and ctx.sig.face.fresh else -math.inf
        W, H = ctx.W, ctx.H
        if face is not None:
            self._zoom = min(max(FACE_W * W / max(face.width, 1.0), ZOOM_MIN), ZOOM_MAX)
            self._face_c = (float(face.center[0]), float(face.center[1]))
        else:
            self._zoom, self._face_c = ZOOM_MIN, (W / 2, H * FACE_AT[1])

    def _length(self, ctx: FxContext) -> float:
        if self.seconds > 0:
            return min(max(self.seconds, MIN_S), MAX_S)
        length = getattr(ctx.sfx, "length", None)
        music = float(length(self.sfx_key)) if length is not None else 0.0
        return min(max(music, MIN_S), MAX_S) if music > 0 else DEFAULT_S

    def finish(self, ctx: FxContext) -> None:
        """Back to live now (the BRB hotkey pressed again)."""
        if self.active and self.state != OUT:
            self._go_out(ctx.now)

    def update(self, ctx: FxContext, out: np.ndarray) -> None:
        now = ctx.now
        t = now - self.t0
        if self._freeze is None:
            self._capture(out)
        face = ctx.sig.face
        if face is not None and face.fresh:
            self._face_t = now
            self._back_since = self._back_since if self._back_since is not None else now
        elif now - self._face_t > 0.3:  # the face model drops a frame now and then
            self._back_since = None

        if self.state == SHOW and t >= self.duration:
            if self.hold_when_away and now - self._face_t > AWAY_S:
                self.state = AWAY
                log.info("brb: you left; holding the card until you're back")
            else:
                self._go_out(now)
        if self.state == AWAY and self._back_since is not None and now - self._back_since >= BACK_S:
            log.info("brb: you're back")
            self._go_out(now)

        if self.state == OUT:
            k = (now - self._t_out) / OUT_S
            if k >= 1.0:
                self.active = False
                return  # `out` already holds the live picture
            # zoom the card back out first, then fade: a fade while zoomed in shows two faces
            e = ease_in_out_sine(min(1.0, k / OUT_ZOOM))
            self._draw(self._buf, t, 1.0 + (self._z_out - 1.0) * (1.0 - e), caption=1.0 - e)
            fade = clamp01((k - (1.0 - OUT_FADE)) / OUT_FADE)
            cv2.addWeighted(self._buf, 1.0 - fade, out, fade, 0, dst=out)
            return
        self._draw(out, t, self._zoom_at(t), caption=1.0)

    def _go_out(self, now: float) -> None:
        self._z_out = self._zoom_at(now - self.t0)
        self.state, self._t_out = OUT, now

    # ------------------------------------------------------------ drawing
    def _capture(self, out: np.ndarray) -> None:
        """Freeze what the meeting sees, graded once: a little punchier, darker at the edges."""
        H, W = out.shape[:2]
        gray = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        punchy = cv2.addWeighted(out, 1.1, gray, -0.1, 0)  # a touch more colour (more turns skin red)
        punchy = cv2.convertScaleAbs(punchy, alpha=1.06, beta=-6)  # a bit more contrast
        self._freeze = cv2.multiply(punchy, _grade_vignette(W, H), scale=1 / 255)
        self._buf = np.empty_like(out)

    def _zoom_at(self, t: float) -> float:
        if t < PUNCH_S:
            return 1.0 + (self._zoom - 1.0) * ease_out_cubic(t / PUNCH_S)
        return self._zoom * (1.0 + CREEP * ease_out_cubic(clamp01((t - PUNCH_S) / CREEP_S)))

    def _draw(self, dst: np.ndarray, t: float, z: float, caption: float) -> None:
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
        if tc > 0 and caption > 0:
            pm, inv3 = _caption_sprite(self.text, W)
            scale = 0.3 + 0.7 * ease_out_back(tc / CAPTION_POP_S, overshoot=2.0)
            blit_scaled(dst, pm, inv3, W / 2, H * CAPTION_Y, scale, min(1.0, tc / 0.06) * caption)


# ---------------------------------------------------------------- sprites
@lru_cache(maxsize=2)
def _grade_vignette(W: int, H: int) -> np.ndarray:
    return vignette_map(W, H, strength=0.35, radius=0.95)


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
