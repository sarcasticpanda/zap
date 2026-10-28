"""Cutaways: your face morphs into a meme's face, the meme plays, then morphs back.

The transition is a gentle zoom dissolve: the camera eases in on your face
while the meme, framed on ITS face at the same size and place, fades in over it
and then settles out to its own framing. Going back is the same in reverse. If a hand hides your face
(the forehead gesture), the last place it was seen is used. Small meme GIFs are
AI-upscaled once in the background (:mod:`zap.effects.enhance`). Meme landmarks
come from :mod:`zap.effects.meme_faces` (background thread, cached on disk);
until they exist, or if a frame has no detectable face, a zoom cross-fade is
used instead.

Memes are rarely 16:9 and often small, so they are shown the way short-video
apps do it: fitted whole (never cropped), upscaled with Lanczos plus a mild
unsharp mask, over a blurred, darkened copy of themselves. ``face_swap =
"cutout"`` puts YOUR live head (hair and all, cut out with the person mask) on
the actor's head, unblended, like a meme head cutout; ``"ai"`` redraws the actor's
face as yours with a face-swap network (:mod:`zap.effects.faceswap`).

Meme files are copyrighted and user-supplied, so a missing one falls back to a
caption card ("*stares at camera*") over a blurred copy of what the meeting was
already seeing (built from ``out``, never the raw camera, so it can't leak the
user while the presence effect has them vanished).
"""
from __future__ import annotations

import logging
import math
import weakref
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from zap.assets import MAX_CLIP_SECONDS, AssetResolver
from zap.effects import enhance, faceswap
from zap.effects.base import Effect, FxContext, Layer
from zap.effects.faceswap import umeyama
from zap.effects.meme_faces import FrameFaces
from zap.effects.morph import FACE_OVAL
from zap.effects.registry import register
from zap.render.blend import blit_premult_centered, clamp01, cover_fit, ease_out_cubic, make_premult
from zap.render.text import TextStyle, render_text, scale_rgba

log = logging.getLogger(__name__)

TRANSITION_IN = 1.1  # seconds: ease in on your face while the meme fades in, then it settles
TRANSITION_OUT = 1.0  # and back
FACE_W_MID = 0.36  # mid-transition both faces are this wide (fraction of the frame width), centred
ZOOM_MAX = 3.0
FADE = 0.5  # the two pictures cross-fade over this middle part of the transition

# Head cutout (face_swap = "cutout"): your head, cut out of the live camera, on the actor's head.
HEAD_SCALE = 1.1  # a meme cutout is a little big for the body
JAW_GROW = 1.04  # the cut follows your jaw line, this much outside it ...
HEAD_GROW = (1.3, 1.45)  # ... and swells to this (x, y) over the top of your head ...
HEAD_UP = 0.15  # ... raised this many face heights for the hair (then trimmed by the person mask)
ALIGN = (33, 133, 362, 263, 1, 61, 291, 152, 10)  # eyes, nose, mouth corners, chin, forehead
HEAD_DIM, HEAD_MAX_GAIN = 0.85, 1.8  # a head darker than this x the actor's face is brightened, up to this
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
        loop: bool = False,
        **params: Any,
    ) -> None:
        super().__init__(name=name, asset=asset, caption=caption, seconds=seconds, cutaway=cutaway,
                         face_swap=face_swap, transition=transition, loop=loop, **params)
        self.loop = bool(loop)  # clips: loop (a repeating gesture) or play once and hold the last frame
        self.cutaway_name = cutaway or name or asset or "cutaway"
        self.asset = asset or self.cutaway_name
        self.caption = caption or f"*{self.cutaway_name.replace('_', ' ')}*"
        self.seconds = max(0.1, float(seconds))
        mode = face_swap if isinstance(face_swap, str) else ("cutout" if face_swap else "")
        self.face_swap = mode if mode in ("cutout", "ai") else ""
        self.needs_mask = self.face_swap == "cutout"  # the head is cut out with the person mask
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
            self._transition(ctx, meme, pts, t / self.t_in, out, "in")
        elif t > self.duration - self.t_out:
            self._transition(ctx, meme, pts, (self.duration - t) / self.t_out, out, "out")
        else:
            np.copyto(out, meme)

    # ------------------------------------------------------------ setup
    def _prepare(self, assets: AssetResolver | None, W: int, H: int) -> None:
        self._size = (W, H)
        self._live = np.empty((H, W, 3), np.uint8)
        self._a = np.empty_like(self._live)
        self._b = np.empty_like(self._live)
        self._prep = _prepared(assets, self.asset, W, H) if assets is not None else None
        if self._prep is not None and self.face_swap == "ai":
            p = self._prep
            faceswap.service().register(p.key, p.frames, p.found_pts, p.faces.ready)
        if self._prep is None:
            self.mode, self.hold = "card", self.seconds
            self._card = _CardBuffers(self.caption, W, H)
        elif self._prep.fps > 0:
            self.mode = "clip"
            clip_len = len(self._prep.frames) / self._prep.fps
            # play the clip through once: it starts during the morph in, the morph out follows its end
            self.hold = max(self.seconds, min(MAX_CLIP_SECONDS, clip_len) - self.t_in)
        else:
            self.mode, self.hold = "image", self.seconds
        self.duration = self.t_in + self.hold + self.t_out

    # ------------------------------------------------------------ drawing
    def _meme_frame(self, ctx: FxContext, t: float) -> tuple[np.ndarray, np.ndarray | None]:
        """The meme frame for time t (clips loop) and its face landmarks; your head/face on it if enabled."""
        prep = self._prep
        n = len(prep.frames)
        i = (int(t * prep.fps) % n if self.loop else min(int(t * prep.fps), n - 1)) if prep.fps > 0 else 0
        pts = prep.faces.get(i)
        if self.face_swap == "ai":
            return (faceswap.service().swapped(prep.key) or prep.frames)[i], pts
        face = ctx.sig.face
        if self.face_swap == "cutout" and pts is not None and face is not None:
            np.copyto(self._a, prep.frames[i])
            paste_head(self._a, pts, ctx.frame, face.lm[:, :2], ctx.person_mask())
            return self._a, pts
        return prep.frames[i], pts

    def _transition(self, ctx: FxContext, meme: np.ndarray, meme_pts: np.ndarray | None, p: float,
                    out: np.ndarray, leg: str) -> None:
        """p = 0: what the meeting saw; p = 1: the meme. Ease in on your face, fade to the meme framed on its
        face, let it settle. Symmetric in p, so the way back (``leg`` "out") is the same played backwards."""
        H, W = out.shape[:2]
        centre = np.float64([W / 2, H / 2])
        face = ctx.sig.face if ctx.sig.face is not None else ctx.last_face
        if face is not None:
            fw_u, c_u = float(face.width), np.asarray(face.center, np.float64)
        else:
            fw_u, c_u = 0.25 * W, centre
        if meme_pts is not None:
            oval = meme_pts[list(FACE_OVAL)]
            fw_m, c_m = float(np.ptp(oval[:, 0])), oval.mean(axis=0).astype(np.float64)
        else:
            fw_m, c_m = 0.25 * W, centre
        p = clamp01(p)
        lo, hi = 0.5 - FADE / 2, 0.5 + FADE / 2
        w = _smooth((p - lo) / FADE)  # how much of the meme shows
        if w < 1.0:
            k = _smooth(p / hi)  # you: easing in on your face until the fade is done
            z = min(max(FACE_W_MID * W / max(fw_u, 1.0), 1.0), ZOOM_MAX)
            cv2.warpAffine(self._live, _compose(z ** k, 0.0, c_u, (1 - k) * c_u + k * centre), (W, H), dst=out,
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if w > 0.0:
            k = 1.0 - _smooth((p - lo) / (1.0 - lo))  # the meme: framed on its face from the fade's start, settling
            z = min(max(FACE_W_MID * W / max(fw_m, 1.0), 1.0), ZOOM_MAX)
            dst = self._b if w < 1.0 else out
            cv2.warpAffine(meme, _compose(z ** k, 0.0, c_m, (1 - k) * c_m + k * centre), (W, H), dst=dst,
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if w < 1.0:
                cv2.addWeighted(out, 1.0 - w, self._b, w, 0, dst=out)

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
            sources, fps = clip.frames[:n], float(clip.fps)
            frames = [present(f, W, H) for f in sources]
        else:
            img = assets.image(key)
            if img is None:
                cache[ck] = None
                return None
            bgr = np.ascontiguousarray(img[..., :3])
            a = img[..., 3]
            if a.min() < 255:  # transparent PNG: composite over black
                bgr = cv2.multiply(bgr, cv2.merge([a, a, a]), scale=1 / 255)
            sources, fps = [bgr], 0.0
            frames = [present(bgr, W, H)]
        tag = f"{key}:{W}x{H}"
        if enhance.available() and sources[0].shape[0] < enhance.SMALL * H:
            ek = enhance.cache_key(sources, tag)
            sharp = enhance.load_cached(ek, len(frames))
            if sharp is not None:
                frames = sharp
            else:  # swapped in place, frame by frame, when done
                need = min(W / sources[0].shape[1], H / sources[0].shape[0])
                enhance.enhancer().submit(ek, sources, lambda f: present(f, W, H), frames, key, need)
        cache[ck] = _Prepared(frames, fps, FrameFaces(frames, tag=tag), key=tag)
    return cache[ck]


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


# ---------------------------------------------------------------- head cutout
def paste_head(dst: np.ndarray, dst_pts: np.ndarray, src: np.ndarray, src_pts: np.ndarray,
               person: np.ndarray | None) -> None:
    """Cut the head at ``src_pts`` out of ``src`` (hair included, via the ``person`` mask when given) and
    paste it over the face at ``dst_pts`` in ``dst``, crisp-edged and unblended, like a meme cutout."""
    oval = src_pts[list(FACE_OVAL)].astype(np.float32)
    c = oval.mean(axis=0)
    fh = float(np.ptp(oval[:, 1]))
    d = oval - c
    # below the eyes the cut follows your jaw line (a hair outside it); above, it swells into a
    # rounded dome over your hair, which the person mask then trims to the hair's real outline
    up = np.clip(-d[:, 1] / (0.5 * fh), 0.0, 1.0)[:, None]
    grow = JAW_GROW + up * (np.float32(HEAD_GROW) - JAW_GROW)
    region = c + d * grow - np.float32([0.0, HEAD_UP * fh]) * up
    sh, sw = src.shape[:2]
    x0, y0 = np.maximum(np.floor(region.min(axis=0)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(region.max(axis=0)).astype(int) + 1, (sw, sh))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return
    mask = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.fillPoly(mask, [np.round((region - (x0, y0)) * 16).astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
    if person is not None:
        mask *= person[y0:y1, x0:x1]
    # your face onto the actor's, a little bigger, about the actor's face centre
    M = umeyama(src_pts[list(ALIGN)], dst_pts[list(ALIGN)]).astype(np.float64)
    dc = dst_pts[list(FACE_OVAL)].mean(axis=0)
    M[:, :2] *= HEAD_SCALE
    M[:, 2] = HEAD_SCALE * M[:, 2] + (1 - HEAD_SCALE) * dc
    corners = np.float64([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ M[:, :2].T + M[:, 2]
    dh, dw = dst.shape[:2]
    u0, v0 = np.maximum(np.floor(corners.min(axis=0)).astype(int), 0)
    u1, v1 = np.minimum(np.ceil(corners.max(axis=0)).astype(int) + 1, (dw, dh))
    if u1 - u0 < 4 or v1 - v0 < 4:
        return
    Mr = M.copy()
    Mr[:, 2] += M[:, :2] @ np.float64([x0, y0]) - (u0, v0)  # ROI to ROI
    size = (int(u1 - u0), int(v1 - v0))
    head = cv2.warpAffine(src[y0:y1, x0:x1], Mr, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a = cv2.warpAffine(mask, Mr, size, flags=cv2.INTER_LINEAR)
    a = np.clip((a - 0.35) * (1 / 0.3), 0.0, 1.0)  # a cut, not a fade: ~1 px of anti-aliasing
    roi = dst[v0:v1, u0:u1]
    inside = (a > 0.5).astype(np.uint8)
    if inside.any():  # a dim webcam head on a lit scene reads as a dark blob: lift it (no mixing)
        mine = float(cv2.mean(cv2.cvtColor(head, cv2.COLOR_BGR2GRAY), inside)[0])
        theirs = float(cv2.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), inside)[0])
        if mine < HEAD_DIM * theirs:
            head = cv2.convertScaleAbs(head, alpha=min(HEAD_MAX_GAIN, HEAD_DIM * theirs / max(mine, 1.0)))
    cv2.blendLinear(head, roi.copy(), a, 1.0 - a, dst=roi)


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
def _compose(s: float, th: float, src_c: np.ndarray, dst_c: np.ndarray) -> np.ndarray:
    """Scale s and rotation th about src_c, moved so src_c lands on dst_c."""
    a, b = s * math.cos(th), s * math.sin(th)
    tx = float(dst_c[0]) - (a * float(src_c[0]) - b * float(src_c[1]))
    ty = float(dst_c[1]) - (b * float(src_c[0]) + a * float(src_c[1]))
    return np.float32([[a, -b, tx], [b, a, ty]])


def _smooth(x: float) -> float:
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
