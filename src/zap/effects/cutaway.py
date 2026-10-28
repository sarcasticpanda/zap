"""Cutaways: your face morphs into a meme's face, the meme plays, then morphs back.

The transition is a real two-image morph (:mod:`zap.effects.morph`): a mesh of
face landmarks on both pictures, eyes sliding onto eyes and jaw onto jaw while
the colours cross-dissolve, so your face *becomes* the meme's. Meme landmarks
come from :mod:`zap.effects.meme_faces` (background thread, cached on disk);
until they exist, or if a frame has no detectable face, a zoom cross-fade is
used instead.

Memes are rarely 16:9 and often small, so they are shown the way short-video
apps do it: fitted whole (never cropped), upscaled with Lanczos plus a mild
unsharp mask, over a blurred, darkened copy of themselves. With ``face_swap``
the clip plays with YOUR face on the actor, redrawn by a face-swap network
(:mod:`zap.effects.faceswap`) once in the background; until that is ready (or
without its models) the original clip plays.

Meme files are copyrighted and user-supplied, so a missing one falls back to a
caption card ("*stares at camera*") over a blurred copy of what the meeting was
already seeing (built from ``out``, never the raw camera, so it can't leak the
user while the presence effect has them vanished).
"""
from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from zap.assets import MAX_CLIP_SECONDS, AssetResolver
from zap.effects import faceswap
from zap.effects.base import Effect, FxContext, Layer
from zap.effects.meme_faces import FrameFaces
from zap.effects.morph import Morph, control_points
from zap.effects.registry import register
from zap.render.blend import blit_premult_centered, clamp01, cover_fit, ease_in_out_sine, ease_out_cubic, make_premult
from zap.render.text import TextStyle, render_text, scale_rgba

log = logging.getLogger(__name__)

TRANSITION_IN = 0.9  # seconds of morph from your face into the meme
TRANSITION_OUT = 0.7  # and back
COLOUR_LAG = 0.2  # the shapes start moving this fraction of the morph before the colours change
CARD_FADE = 0.25
CARD_PUNCH = 0.04
CARD_DOWNSCALE = 8  # the card background is blurred at 1/8 resolution
CARD_FONT = "georgiai.ttf"
BACKDROP_DIM = 0.6  # brightness of the blurred backdrop behind a fitted meme
EDGE_FEATHER = 0.1  # the meme fades into the backdrop over this fraction of its size
SHARPEN_ABOVE = 1.3  # upscale factor above which a fitted meme gets an unsharp mask
NEAR_16_9 = 0.12  # aspect mismatch under which a meme simply fills the frame

@dataclass
class _Prepared:
    frames: list[np.ndarray]  # presentation frames, BGR at output size
    fps: float  # 0.0 for a still
    faces: FrameFaces
    key: str = ""  # identifies these frames (asset and size) for the face-swap service

    def found_pts(self, i: int) -> np.ndarray | None:
        """Landmarks of frame i only if the face model really saw a face there."""
        return self.faces.pts[i] if self.faces.found[i] else None


# Prepared memes, per resolver, so a fresh resolver (config reload) starts clean.
_FITTED: weakref.WeakKeyDictionary[AssetResolver, dict[tuple[str, int, int], _Prepared | None]] = (
    weakref.WeakKeyDictionary()
)


@register("cutaway")
class Cutaway(Effect):
    """Morph into ``asset`` (clip, else image, else a caption card), hold ``seconds``, morph back.

    ``name`` is the cutaway's config name (``jim``); the effect's registry name
    stays ``"cutaway"``. ``cutaway=`` is accepted as an alias for ``name``.
    """

    layer = Layer.CUTAWAY
    needs_face = True  # the morph and the face swap follow your face every frame

    def __init__(
        self,
        name: str = "",
        asset: str = "",
        caption: str = "",
        seconds: float = 2.5,
        cutaway: str = "",
        face_swap: bool = False,
        transition: bool = True,
        **params: Any,
    ) -> None:
        super().__init__(name=name, asset=asset, caption=caption, seconds=seconds, cutaway=cutaway,
                         face_swap=face_swap, transition=transition, **params)
        self.cutaway_name = cutaway or name or asset or "cutaway"
        self.asset = asset or self.cutaway_name
        self.caption = caption or f"*{self.cutaway_name.replace('_', ' ')}*"
        self.seconds = max(0.1, float(seconds))
        self.face_swap = bool(face_swap)
        self.t_in = TRANSITION_IN if transition else 0.0
        self.t_out = TRANSITION_OUT if transition else 0.0
        self.hold = self.seconds
        self.duration = self.t_in + self.hold + self.t_out
        self.mode = "card"  # "clip" | "image" | "card", decided in _prepare()
        self._prep: _Prepared | None = None
        self._card: _CardBuffers | None = None
        self._size = (0, 0)

    @classmethod
    def prewarm(cls, width: int, height: int, assets: AssetResolver | None,
                cutaways: dict[str, dict[str, Any]]) -> None:
        """Decode, fit and start face-finding for every configured cutaway so the first trigger doesn't hitch."""
        for name, section in cutaways.items():
            try:
                cls(cutaway=name, **section)._prepare(assets, width, height)
            except Exception:
                log.exception("could not prepare cutaway %s", name)

    @property
    def sfx(self) -> str:
        """The sound key played on start; SfxPlayer falls back to "cutaway"."""
        return f"cutaway_{self.cutaway_name}"

    # ------------------------------------------------------------ lifecycle
    def start(self, ctx: FxContext, **kw: Any) -> None:
        super().start(ctx, **kw)
        ctx.sfx.play(self.sfx)
        self._prepare(ctx.assets, ctx.W, ctx.H)

    def update(self, ctx: FxContext, out: np.ndarray) -> None:
        t = self.elapsed(ctx)
        if t >= self.duration:
            self.active = False
            return
        H, W = out.shape[:2]
        if self._size != (W, H):
            self._prepare(ctx.assets, W, H)
        np.copyto(self._live, out)  # what the meeting was seeing (lower layers)
        if self.mode == "card":
            self._draw_card(t, out)
            return
        meme, pts = self._meme_frame(ctx, t)
        if t < self.t_in:
            self._morph(ctx, meme, pts, t / self.t_in, out)
        elif t > self.duration - self.t_out:
            self._morph(ctx, meme, pts, (self.duration - t) / self.t_out, out)
        else:
            np.copyto(out, meme)

    # ------------------------------------------------------------ setup
    def _prepare(self, assets: AssetResolver | None, W: int, H: int) -> None:
        self._size = (W, H)
        self._live = np.empty((H, W, 3), np.uint8)
        self._a = np.empty_like(self._live)
        self._b = np.empty_like(self._live)
        self._prep = _prepared(assets, self.asset, W, H) if assets is not None else None
        self._morpher = _morpher(W, H)
        if self._prep is not None and self.face_swap:
            p = self._prep
            faceswap.service().register(p.key, p.frames, p.found_pts, p.faces.ready)
        if self._prep is None:
            self.mode, self.hold = "card", self.seconds
            self._card = _CardBuffers(self.caption, W, H)
        elif self._prep.fps > 0:
            self.mode = "clip"
            clip_len = len(self._prep.frames) / self._prep.fps
            self.hold = max(self.seconds, min(MAX_CLIP_SECONDS, clip_len) - self.t_in - self.t_out)
        else:
            self.mode, self.hold = "image", self.seconds
        self.duration = self.t_in + self.hold + self.t_out

    # ------------------------------------------------------------ drawing
    def _meme_frame(self, ctx: FxContext, t: float) -> tuple[np.ndarray, np.ndarray | None]:
        """The meme frame for time t (clips loop) and its face landmarks; your face on it if swapped."""
        prep = self._prep
        i = int(t * prep.fps) % len(prep.frames) if prep.fps > 0 else 0
        frames = (faceswap.service().swapped(prep.key) if self.face_swap else None) or prep.frames
        return frames[i], prep.faces.get(i)

    def _morph(self, ctx: FxContext, meme: np.ndarray, meme_pts: np.ndarray | None, p: float,
               out: np.ndarray) -> None:
        """p = 0: what the meeting saw; p = 1: the meme. Your face turns into the meme's in between."""
        p = clamp01(p)
        shape = ease_in_out_sine(p)
        colour = _smoothstep((p - COLOUR_LAG) / (1.0 - 2 * COLOUR_LAG))
        H, W = out.shape[:2]
        face = ctx.sig.face
        if face is not None and meme_pts is not None:
            self._morpher.render(self._live, control_points(face.lm), meme, control_points(meme_pts),
                                 shape, out, alpha=colour)
            return
        # no faces to morph between: zoom cross-fade
        cx, cy = (face.center if face is not None else (W / 2, H / 2))
        cv2.warpAffine(self._live, _scale_about(1.0 + 0.25 * shape, cx, cy), (W, H), dst=self._b,
                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        cv2.warpAffine(meme, _scale_about(1.15 - 0.15 * shape, W / 2, H / 2), (W, H), dst=out,
                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        cv2.addWeighted(out, colour, self._b, 1.0 - colour, 0, dst=out)

    def _draw_card(self, t: float, out: np.ndarray) -> None:
        self._card.draw(out, 1.0 + CARD_PUNCH * ease_out_cubic(t / self.duration))
        a = min(1.0, t / CARD_FADE, (self.duration - t) / CARD_FADE) if CARD_FADE > 0 else 1.0
        if a < 1.0:
            cv2.addWeighted(out, max(0.0, a), self._live, 1.0 - max(0.0, a), 0, dst=out)


# ---------------------------------------------------------------- preparing memes
def _prepared(assets: AssetResolver, key: str, W: int, H: int) -> _Prepared | None:
    cache = _FITTED.setdefault(assets, {})
    ck = (key, W, H)
    if ck not in cache:
        clip = assets.clip(key)
        if clip is not None and clip.frames:
            n = max(1, int(MAX_CLIP_SECONDS * clip.fps))
            frames, fps = [present(f, W, H) for f in clip.frames[:n]], float(clip.fps)
        else:
            img = assets.image(key)
            if img is None:
                cache[ck] = None
                return None
            bgr = np.ascontiguousarray(img[..., :3])
            a = img[..., 3]
            if a.min() < 255:  # transparent PNG: composite over black
                bgr = cv2.multiply(bgr, cv2.merge([a, a, a]), scale=1 / 255)
            frames, fps = [present(bgr, W, H)], 0.0
        tag = f"{key}:{W}x{H}"
        cache[ck] = _Prepared(frames, fps, FrameFaces(frames, tag=tag), key=tag)
    return cache[ck]


_MORPHERS: dict[tuple[int, int], Morph] = {}


def _morpher(W: int, H: int) -> Morph:
    if (W, H) not in _MORPHERS:
        _MORPHERS[(W, H)] = Morph(W, H)
    return _MORPHERS[(W, H)]


def present(src: np.ndarray, W: int, H: int) -> np.ndarray:
    """Fit a meme into W x H uncropped, sharpened when upscaled, over a blurred copy of itself."""
    ih, iw = src.shape[:2]
    if abs(iw / ih - W / H) / (W / H) < NEAR_16_9:
        s = max(W / iw, H / ih)
        out = cover_fit(src, W, H)
        return _sharpen(out, s) if s > SHARPEN_ABOVE else out
    s = min(W / iw, H / ih)
    nw, nh = max(1, round(iw * s)), max(1, round(ih * s))
    fg = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_LANCZOS4 if s > 1 else cv2.INTER_AREA)
    if s > SHARPEN_ABOVE:
        fg = _sharpen(fg, s)
    small = cv2.GaussianBlur(cover_fit(src, max(16, W // 10), max(9, H // 10)), (0, 0), 2.5)
    out = cv2.convertScaleAbs(cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR), alpha=BACKDROP_DIM)
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    # feather the edges that meet the backdrop: no hard frame line (it would wobble through a morph)
    f = max(2.0, EDGE_FEATHER * min(nw, nh))
    ax = np.clip(np.minimum(np.arange(nw), nw - 1 - np.arange(nw)) / f, 0, 1) if x0 > 0 else np.ones(nw)
    ay = np.clip(np.minimum(np.arange(nh), nh - 1 - np.arange(nh)) / f, 0, 1) if y0 > 0 else np.ones(nh)
    a = (ay[:, None] * ax[None, :]).astype(np.float32)
    a = a * a * (3 - 2 * a)
    roi = out[y0:y0 + nh, x0:x0 + nw]
    cv2.blendLinear(fg, roi.copy(), a, 1.0 - a, dst=roi)
    return out


def _sharpen(img: np.ndarray, upscale: float) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), 0.6 + 0.35 * min(upscale, 4.0))
    return cv2.addWeighted(img, 1.7, blur, -0.7, 0)


# ---------------------------------------------------------------- caption card
class _CardBuffers:
    """Preallocated buffers and sprites for the caption card at one output size."""

    def __init__(self, caption: str, W: int, H: int) -> None:
        sw, sh = max(8, W // CARD_DOWNSCALE), max(8, H // CARD_DOWNSCALE)
        self.small = np.empty((sh, sw, 3), np.uint8)
        self.small2 = np.empty_like(self.small)
        self.mul, self.glow = _card_maps(W, H)
        self.pm, self.inv3 = _caption_sprite(caption, W)

    def draw(self, out: np.ndarray, s: float) -> None:
        H, W = out.shape[:2]
        sh, sw = self.small.shape[:2]
        # area-average down (no shimmer as the person moves), blur, punch in at low res, then up
        cv2.resize(out, (sw, sh), dst=self.small, interpolation=cv2.INTER_AREA)
        cv2.GaussianBlur(self.small, (0, 0), 2.2, dst=self.small)
        cv2.warpAffine(self.small, _scale_about(s, sw / 2, sh / 2), (sw, sh), dst=self.small2,
                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        cv2.resize(self.small2, (W, H), dst=out, interpolation=cv2.INTER_LINEAR)
        cv2.multiply(out, self.mul, dst=out, scale=1 / 255)
        cv2.add(out, self.glow, dst=out)
        blit_scaled(out, self.pm, self.inv3, W / 2, H * 0.5, s)


@lru_cache(maxsize=4)
def _card_maps(W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """(darkening multiplier, cool glow) maps for a dark radial-gradient card."""
    xs = np.linspace(-1, 1, W, dtype=np.float32)
    ys = np.linspace(-1, 1, H, dtype=np.float32)
    d = np.clip(np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2) / np.sqrt(2), 0, 1)
    centre = (1.0 - d) ** 1.6
    mul = (0.07 + 0.30 * centre) * 255
    m8 = mul.astype(np.uint8)
    glow = np.stack([centre * 34, centre * 24, centre * 20], axis=-1).astype(np.uint8)  # BGR: slate blue
    return cv2.merge([m8, m8, m8]), np.ascontiguousarray(glow)


@lru_cache(maxsize=16)
def _caption_sprite(caption: str, W: int) -> tuple[np.ndarray, np.ndarray]:
    k = W / 1280
    style = TextStyle(font=CARD_FONT, size=max(16, int(84 * k)), color=(255, 255, 255, 255),
                      shadow=max(2, int(8 * k)), shadow_alpha=0.75)
    rgba = render_text(caption, style)
    max_w = 0.86 * W
    if rgba.shape[1] > max_w:
        rgba = scale_rgba(rgba, max_w / rgba.shape[1])
    return make_premult(rgba)


# ---------------------------------------------------------------- helpers
def _smoothstep(x: float) -> float:
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def _scale_about(s: float, cx: float, cy: float) -> np.ndarray:
    return np.float32([[s, 0, cx - s * cx], [0, s, cy - s * cy]])


def blit_scaled(dst: np.ndarray, pm: np.ndarray, inv3: np.ndarray, cx: float, cy: float,
                scale: float = 1.0, opacity: float = 1.0) -> None:
    """Centred premultiplied blit with a scale and opacity (resizes the small sprite, not the frame)."""
    opacity = clamp01(opacity)
    if scale <= 0.01 or opacity <= 0.004:
        return
    if abs(scale - 1.0) > 1e-3:
        size = (max(1, round(pm.shape[1] * scale)), max(1, round(pm.shape[0] * scale)))
        pm = cv2.resize(pm, size, interpolation=cv2.INTER_LINEAR)
        inv3 = cv2.resize(inv3, size, interpolation=cv2.INTER_LINEAR)
    if opacity < 0.996:
        pm = cv2.convertScaleAbs(pm, alpha=opacity)
        inv3 = cv2.convertScaleAbs(inv3, alpha=opacity, beta=255 * (1 - opacity))
    blit_premult_centered(dst, pm, inv3, cx, cy)
