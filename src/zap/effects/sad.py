"""Sad mode: a crying-Jordan face on yours plus a rainy, blue, slow-zoom grade.

Covering your face and dropping your hands reveals you "crying". With the
user's ``crying_jordan`` cutout (never bundled), the meme face is warped onto
yours by a smoothed similarity transform fitted to a few stable landmarks. The
meme's own landmarks are found once, off the render thread, and cached on disk;
until they arrive (or if the picture has no detectable face) the cutout is
aligned by its alpha bounding box instead. Hands stay in front of the meme.

The procedural grade always runs, with or without the cutout: 70% desaturation
with a cool tint, scrolling rain, tears from the lower eyelids, a slow push-in
and a sad-violin caption. Every full-frame step is one cv2 call into a
preallocated buffer; tears are stamped with a single numpy gather/scatter.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import weakref
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from zap import paths
from zap.assets import AssetResolver
from zap.effects.base import Effect, FxContext, Layer
from zap.effects.morph import FACE_OVAL
from zap.effects.registry import register
from zap.fetch_models import model_path
from zap.render.blend import blit_premult, ease_in_out_sine, make_premult, vignette_map
from zap.render.text import TextStyle, render_emoji, render_text
from zap.signals import (
    FACE_EYE_CORNERS,
    FACE_LEFT_EDGE,
    FACE_LOWER_LIDS,
    FACE_NOSE_BRIDGE,
    FACE_RIGHT_EDGE,
    MIDDLE_MCP,
    WRIST,
)
from zap.vision.geometry import OneEuro

log = logging.getLogger(__name__)

MEME_KEY = "crying_jordan"
ANCHORS = (33, 133, 362, 263, 1, 61, 291, 152, 10)  # eye corners, nose tip, mouth corners, chin, forehead
MEME_MAX_SIDE = 720  # the warped face never needs more pixels than this
CACHE_VERSION = 1
HOLD_S = 0.3  # keep the meme where it was this long after the face is lost

SECONDS = 6.0
FACE_IN_S = 0.7  # your face turns into the meme's over this long
GRADE_IN_S = 0.45  # the rain and blue grade fade in
COVER_SCALE = 1.12  # a little bigger than your face so glasses and cheeks don't peek out
# Your hair and ears around the meme face are replaced by the room (clean plate), so only
# the meme's head sits on your shoulders: the area is the face outline scaled (x, y) about
# its centre and moved up to take in the hair, intersected with the person mask.
HIDE_HEAD = (1.5, 1.45)
HIDE_UP = 0.2  # face heights
HIDE_MIN_PLATE = 0.6  # plate coverage needed (else the hidden area would be a guess)
FADE_S = 0.4
ZOOM = 0.08

DESAT = 0.7
TINT_BGR = (1.10, 1.0, 0.86)  # cool: blue up, red down
LIFT_BGR = (10.0, 4.0, 0.0)  # bluish blacks
DARKEN = 0.9

RAIN_STREAKS = 600  # at 1280x720; scaled by area
RAIN_SPEED = 900.0  # px/s along the slant
RAIN_SLANT_DEG = 10.0

TEAR_RATE = 6.0  # per eye per second
TEAR_SPEED = 120.0  # px/s for a 250 px wide face
TEAR_BGR = (255, 220, 180)
TEAR_ALPHA = 0.7
MAX_TEARS = 64

CAPTION = "*sad violin plays*"
CAPTION_FONT = "georgiai.ttf"
LOUDLY_CRYING = "\U0001F62D"

_K3 = np.ones((3, 3), np.uint8)


# ================================================================ meme face
@dataclass
class MemeFace:
    """A meme cutout prepared for warping; ``lm`` arrives later from a background thread."""

    sha: str
    bgra: np.ndarray  # cropped to the alpha bbox, at most MEME_MAX_SIDE, colour bled under alpha 0
    origin: tuple[int, int]  # crop offset in the source image
    scale: float  # working pixels per source pixel
    lab_mean: np.ndarray
    lab_std: np.ndarray
    lm: np.ndarray | None = None  # (478, 2) working coordinates; None until/unless a face is found
    ready: threading.Event = field(default_factory=threading.Event)

    @property
    def anchors(self) -> np.ndarray | None:
        lm = self.lm
        return None if lm is None else lm[list(ANCHORS)]

    def to_source(self, pts: np.ndarray) -> np.ndarray:
        return pts / self.scale + np.float32(self.origin)

    def from_source(self, pts: np.ndarray) -> np.ndarray:
        return ((pts - np.float32(self.origin)) * self.scale).astype(np.float32)

    def set_landmarks(self, lm: np.ndarray, opaque: bool) -> None:
        if opaque:  # no cutout alpha: cut the face out along its landmarks instead
            bgra = self.bgra.copy()
            hull = cv2.convexHull(_expand(lm, 1.06).astype(np.int32))
            a = np.zeros(bgra.shape[:2], np.uint8)
            cv2.fillConvexPoly(a, hull, 255, cv2.LINE_AA)
            bgra[..., 3] = cv2.GaussianBlur(a, (0, 0), max(1.0, bgra.shape[1] / 120))
            self.bgra = bgra
        hull_mask = np.zeros(self.bgra.shape[:2], np.uint8)
        cv2.fillConvexPoly(hull_mask, cv2.convexHull(lm.astype(np.int32)), 255)
        self.lab_mean, self.lab_std = _lab_stats(self.bgra, (self.bgra[..., 3] > 200) & (hull_mask > 0))
        self.lm = lm


_MEMES: dict[str, MemeFace] = {}
_BY_ID: dict[int, tuple[weakref.ref, MemeFace]] = {}
_LOCK = threading.Lock()
_MISS = object()


def meme_face(img: np.ndarray) -> MemeFace:
    """Prepare a BGRA cutout once and start finding its landmarks on a background thread."""
    hit = _BY_ID.get(id(img))
    if hit is not None and hit[0]() is img:
        return hit[1]
    sha = hashlib.sha1(np.ascontiguousarray(img).tobytes() + repr(img.shape).encode()).hexdigest()
    with _LOCK:
        mf = _MEMES.get(sha)
        if mf is None:
            mf, opaque = _prepare_meme(img, sha)
            _MEMES[sha] = mf
            threading.Thread(target=_find_landmarks, args=(mf, opaque), name="zap-meme-landmarks",
                             daemon=True).start()
        _BY_ID[id(img)] = (weakref.ref(img), mf)
    return mf


def _prepare_meme(img: np.ndarray, sha: str) -> tuple[MemeFace, bool]:
    ys, xs = np.nonzero(img[..., 3] > 8)
    if len(xs):
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    else:
        x0, y0, x1, y1 = 0, 0, img.shape[1], img.shape[0]
    crop = img[y0:y1, x0:x1]
    s = min(1.0, MEME_MAX_SIDE / max(crop.shape[:2]))
    if s < 1.0:
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    crop = np.ascontiguousarray(crop)
    opaque = bool(crop[..., 3].min() == 255)
    if opaque:  # until landmarks arrive, an inscribed feathered ellipse
        h, w = crop.shape[:2]
        a = np.zeros((h, w), np.uint8)
        cv2.ellipse(a, (w // 2, h // 2), (int(w * 0.42), int(h * 0.46)), 0, 0, 360, 255, -1, cv2.LINE_AA)
        crop[..., 3] = cv2.GaussianBlur(a, (0, 0), max(1.0, w / 60))
    crop = _bleed(crop)
    mean, std = _lab_stats(crop, crop[..., 3] > 200)
    return MemeFace(sha, crop, (x0, y0), s, mean, std), opaque


def _bleed(bgra: np.ndarray) -> np.ndarray:
    """Spread edge colours under transparent pixels so bilinear warping leaves no dark fringe."""
    a = bgra[..., 3]
    if a.min() == 255:
        return bgra
    m = (a > 128).astype(np.float32)
    k = max(3, (min(a.shape) // 12) | 1)
    num = cv2.GaussianBlur(bgra[..., :3].astype(np.float32) * m[..., None], (k, k), 0)
    den = cv2.GaussianBlur(m, (k, k), 0)
    fill = np.clip(num / np.maximum(den, 1e-3)[..., None], 0, 255).astype(np.uint8)
    out = bgra.copy()
    hole = a <= 128
    out[..., :3][hole] = fill[hole]
    return out


def _lab_stats(bgra: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(np.ascontiguousarray(bgra[..., :3]), cv2.COLOR_BGR2LAB)
    m8 = mask.astype(np.uint8)
    mean, std = cv2.meanStdDev(lab, mask=m8 if m8.any() else None)
    return mean.ravel().astype(np.float32), std.ravel().astype(np.float32)


def _cache_path(sha: str) -> Path:
    return paths.calibration_dir() / f"meme_landmarks_{sha[:12]}.json"


def _load_cached(sha: str) -> Any:
    try:
        data = json.loads(_cache_path(sha).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _MISS
    except (OSError, ValueError) as e:
        log.warning("ignoring unreadable meme landmark cache: %s", e)
        return _MISS
    if not isinstance(data, dict) or data.get("version") != CACHE_VERSION or data.get("sha1") != sha:
        return _MISS
    if data.get("landmarks") is None:
        return None
    lm = np.asarray(data["landmarks"], np.float32)
    return lm if lm.shape == (478, 2) else _MISS


def _save_cached(sha: str, lm: np.ndarray | None) -> None:
    p = _cache_path(sha)
    data = {"version": CACHE_VERSION, "sha1": sha,
            "landmarks": None if lm is None else np.round(lm, 2).tolist()}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data), encoding="utf-8")
    except OSError as e:
        log.warning("could not cache meme landmarks to %s: %s", p, e)


def _find_landmarks(mf: MemeFace, opaque: bool) -> None:
    try:
        src = _load_cached(mf.sha)
        if src is _MISS:
            found = _detect(mf.bgra)
            if found is _MISS:
                return  # model unavailable: don't cache, try again next run
            src = None if found is None else mf.to_source(found)
            _save_cached(mf.sha, src)
        if src is None:
            log.info("no face found in %s; aligning it by its outline instead", MEME_KEY)
        else:
            mf.set_landmarks(mf.from_source(src), opaque)
    except Exception:
        log.exception("finding the %s landmarks failed; aligning it by its outline instead", MEME_KEY)
    finally:
        mf.ready.set()


def _detect(bgra: np.ndarray) -> Any:
    """Face landmarks of a cutout in its own pixels, None if no face, _MISS if the model can't run."""
    p = model_path("face_landmarker.task")
    if not p.is_file():
        log.warning("face model missing (%s); run the model download first", p.name)
        return _MISS
    os.environ.setdefault("GLOG_minloglevel", "2")
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    a = bgra[..., 3:4].astype(np.float32) / 255
    comp = (bgra[..., :3].astype(np.float32) * a + 128 * (1 - a)).astype(np.uint8)  # over mid-grey
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=p.read_bytes()),  # buffer: non-ASCII paths are fine
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(opts)
    try:
        for pad_frac in (0.3, 0.0):  # a cutout fills its picture; padding helps the face detector
            pad = int(pad_frac * max(comp.shape[:2]))
            img = cv2.copyMakeBorder(comp, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(128, 128, 128))
            rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if res.face_landmarks:
                pts = np.array([[q.x, q.y] for q in res.face_landmarks[0]], np.float32)
                return pts * np.float32([img.shape[1], img.shape[0]]) - pad
        return None
    finally:
        landmarker.close()


# ================================================================ effect
@register("sad")
class SadMode(Effect):
    """Crying-Jordan face (when the cutout exists) over the procedural sad grade, for 6 s + 0.4 s fade."""

    layer = Layer.LOOK
    needs_face = True
    needs_mask = True  # hiding your hair needs the person mask every frame
    sfx_key = "sad"

    def __init__(self, seconds: float = SECONDS, fade_s: float = FADE_S, seed: int | None = None,
                 **params: Any) -> None:
        super().__init__(seconds=seconds, fade_s=fade_s, seed=seed, **params)
        self.seconds = max(0.1, float(seconds))
        self.fade_s = max(0.0, float(fade_s))
        self.meme: MemeFace | None = None
        self._rng = np.random.default_rng(seed)
        self._tears = _Tears(self._rng)
        self._align = _Aligner()
        self._tone = _ToneMatch()
        self._size = (0, 0)
        self._zoom_c = (0.0, 0.0)
        self._last_now = 0.0
        self._toned: np.ndarray | None = None  # meme BGRA with the current tone LUT applied
        self._toned_key: tuple[int, np.ndarray | None] | None = None

    @classmethod
    def prewarm(cls, width: int, height: int, assets: AssetResolver | None = None) -> None:
        """Build textures/sprites and start the meme landmark search so the first trigger doesn't hitch."""
        _rain_texture(width, height)
        _vignette(width, height)
        _caption_sprite(width)
        _emoji_sprite(width)
        if assets is not None:
            img = assets.image(MEME_KEY)
            if img is not None:
                meme_face(img)

    def start(self, ctx: FxContext, **kw: Any) -> None:
        super().start(ctx, **kw)
        self._alloc(ctx.W, ctx.H)
        img = ctx.assets.image(MEME_KEY) if ctx.assets is not None else None
        self.meme = meme_face(img) if img is not None else None
        face = ctx.sig.face
        self._zoom_c = (float(face.center[0]), float(face.center[1])) if face is not None else (ctx.W / 2, ctx.H / 2)
        self._tears.reset()
        self._align.reset()
        self._tone.reset()
        self._toned = self._toned_key = None
        self._last_now = ctx.now

    def _alloc(self, W: int, H: int) -> None:
        self._size = (W, H)
        self._buf = np.empty((H, W, 3), np.uint8)
        self._orig = np.empty_like(self._buf)
        self._rain = _rain_texture(W, H)
        self._vig = _vignette(W, H)
        self._grade = _grade_matrix()
        self._caption = _caption_sprite(W)
        self._emoji = _emoji_sprite(W)

    # ------------------------------------------------------------ render
    def update(self, ctx: FxContext, out: np.ndarray) -> None:
        t = self.elapsed(ctx)
        if t >= self.seconds + self.fade_s:
            self.active = False
            return
        H, W = out.shape[:2]
        if self._size != (W, H):
            self._alloc(W, H)
        dt = min(0.1, max(0.0, ctx.now - self._last_now))
        self._last_now = ctx.now
        fading = t > self.seconds and self.fade_s > 0
        entering = t < GRADE_IN_S
        if fading or entering:
            np.copyto(self._orig, out)

        s = 1.0 + ZOOM * ease_in_out_sine(t / self.seconds)
        (x0, y0, cw, ch), Z = _zoom_crop(W, H, s, *self._zoom_c)
        cv2.resize(out[y0:y0 + ch, x0:x0 + cw], (W, H), dst=self._buf, interpolation=cv2.INTER_LINEAR)
        face = ctx.sig.face
        face_pts = _apply(Z, face.lm[:, :2]) if face is not None else None
        if self.meme is not None:
            k = ease_in_out_sine(min(1.0, t / FACE_IN_S))
            if face_pts is not None:
                self._hide_head(ctx, self._buf, Z, face_pts, k)
            self._draw_meme(ctx, self._buf, Z, face_pts, k)

        cv2.transform(self._buf, self._grade, dst=out)  # desaturate + cool tint + lift
        oy, ox = _rain_offset(t, W, H)
        cv2.add(out, self._rain[oy:oy + H, ox:ox + W], dst=out)
        fw = face.width * s if face is not None else 0.0
        self._tears.step(dt, face_pts, fw)
        self._tears.draw(out)
        cv2.multiply(out, self._vig, dst=out, scale=1 / 255)  # vignette + 10% darker

        k = W / 1280
        pm, inv3 = self._caption
        blit_premult(out, pm, inv3, (W - pm.shape[1]) // 2, int(H - pm.shape[0] - 34 * k))
        if self.meme is None:
            pm, inv3 = self._emoji
            bob = 6 * k * math.sin(2 * math.pi * 0.8 * t)
            blit_premult(out, pm, inv3, int(W - pm.shape[1] - 40 * k), int(36 * k + bob))

        if fading or entering:
            a = 1.0 - (t - self.seconds) / self.fade_s if fading else ease_in_out_sine(t / GRADE_IN_S)
            cv2.addWeighted(out, a, self._orig, 1.0 - a, 0, dst=out)

    def _hide_head(self, ctx: FxContext, buf: np.ndarray, Z: np.ndarray, face_pts: np.ndarray, k: float) -> None:
        """Paint the room over your head around the face (hair, ears), hands kept, faded in by ``k``."""
        plate, person = ctx.plate, ctx.person_mask()
        if plate is None or person is None or k <= 0.01 or plate.coverage < HIDE_MIN_PLATE:
            return
        oval = face_pts[list(FACE_OVAL)]
        c = oval.mean(axis=0)
        fh = float(np.ptp(oval[:, 1]))
        region = (oval - c) * np.float32(HIDE_HEAD) + c - np.float32([0.0, HIDE_UP * fh])
        H, W = buf.shape[:2]
        x0, y0 = np.maximum(region.min(axis=0).astype(int) - 8, 0)
        x1, y1 = np.minimum(region.max(axis=0).astype(int) + 8, (W, H))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return
        Zr = Z.astype(np.float32).copy()
        Zr[:, 2] -= (x0, y0)
        size = (int(x1 - x0), int(y1 - y0))
        room = cv2.warpAffine(plate.image(), Zr, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        you = cv2.warpAffine(person, Zr, size, flags=cv2.INTER_LINEAR)
        area = np.zeros((size[1], size[0]), np.uint8)
        cv2.fillConvexPoly(area, cv2.convexHull((region - (x0, y0)).astype(np.int32)), 255, cv2.LINE_AA)
        for hand in ctx.sig.hands:
            _cut_hand(area, _apply(Z, hand.lm[:, :2]) - np.float32([x0, y0]))
        b = max(3, int(0.08 * fh) | 1)
        area = cv2.GaussianBlur(area, (b, b), 0)
        a = cv2.GaussianBlur(you, (5, 5), 0) * area.astype(np.float32) * (k / 255)
        roi = buf[y0:y1, x0:x1]
        cv2.blendLinear(room, roi.copy(), a, 1.0 - a, dst=roi)

    def _draw_meme(self, ctx: FxContext, buf: np.ndarray, Z: np.ndarray, face_pts: np.ndarray | None,
                   k: float = 1.0) -> None:
        """Warp the meme face onto the (zoomed) face in `buf`, only inside its own ROI, hands in front.

        ``k`` (0..1) is the face transition: the meme grows from 80% and fades in over FACE_IN_S.
        """
        mf = self.meme
        if face_pts is not None:
            self._tone.observe(buf, face_pts, mf, ctx.now)
            M = self._align.update(mf, face_pts, ctx.now)
        else:
            M = self._align.hold(ctx.now)
        if M is None or k <= 0.01:
            return
        if k < 1.0 and face_pts is not None:  # grow onto the face from its centre
            c = face_pts[FACE_NOSE_BRIDGE]
            g = 0.8 + 0.2 * k
            M = np.float32([[g, 0, c[0] - g * c[0]], [0, g, c[1] - g * c[1]]]) @ np.vstack([M, [0, 0, 1]])
        bgra = self._toned_meme(mf)
        h, w = bgra.shape[:2]
        H, W = buf.shape[:2]
        corners = _apply(M, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
        x0, y0 = max(0, int(corners[:, 0].min())), max(0, int(corners[:, 1].min()))
        x1, y1 = min(W, math.ceil(corners[:, 0].max())), min(H, math.ceil(corners[:, 1].max()))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        Mr = M.copy()
        Mr[:, 2] -= (x0, y0)
        patch = cv2.warpAffine(bgra, Mr, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        alpha = cv2.extractChannel(patch, 3)
        for hand in ctx.sig.hands:
            _cut_hand(alpha, _apply(Z, hand.lm[:, :2]) - np.float32([x0, y0]))
        cv2.erode(alpha, _K3, dst=alpha)
        cv2.GaussianBlur(alpha, (7, 7), 0, dst=alpha)
        if k < 1.0:
            cv2.convertScaleAbs(alpha, dst=alpha, alpha=k)
        _blend_u8(buf[y0:y1, x0:x1], cv2.cvtColor(patch, cv2.COLOR_BGRA2BGR), alpha)

    def _toned_meme(self, mf: MemeFace) -> np.ndarray:
        """The meme with the tone LUT applied, redone only when the LUT moves (LAB at ROI size is ~2.6 ms)."""
        bgra, lut = mf.bgra, self._tone.lut
        if lut is None:
            return bgra
        key = self._toned_key
        stale = (key is None or key[0] != id(bgra) or key[1] is None
                 or int(np.abs(lut.astype(np.int16) - key[1]).max()) > 2)
        if stale or self._toned is None:
            lab = cv2.cvtColor(np.ascontiguousarray(bgra[..., :3]), cv2.COLOR_BGR2LAB)
            cv2.LUT(lab, lut, dst=lab)
            toned = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2BGRA)
            toned[..., 3] = bgra[..., 3]
            self._toned, self._toned_key = toned, (id(bgra), lut.astype(np.int16))
        return self._toned


# ================================================================ alignment
class _Aligner:
    """Meme -> live similarity transform, One-Euro smoothed as (anchor position, log scale, angle)."""

    def __init__(self) -> None:
        self._pos = OneEuro(min_cutoff=1.5, beta=0.06)  # responsive: the meme must stay on a moving head
        self._rot = OneEuro(min_cutoff=1.0, beta=0.3)
        self.M: np.ndarray | None = None
        self._t = -math.inf

    def reset(self) -> None:
        self._pos.reset()
        self._rot.reset()
        self.M, self._t = None, -math.inf

    def update(self, mf: MemeFace, face_pts: np.ndarray, now: float) -> np.ndarray | None:
        raw = _similarity(mf, face_pts)
        if raw is None:
            return self.hold(now)
        a, b = float(raw[0, 0]), float(raw[1, 0])
        h, w = mf.bgra.shape[:2]
        c = np.array([w / 2, h / 2])
        cd = self._pos(raw[:, :2].astype(np.float64) @ c + raw[:, 2], now)
        log_s, th = self._rot([math.log(math.hypot(a, b)), math.atan2(b, a)], now)
        s = math.exp(log_s) * COVER_SCALE
        a, b = s * math.cos(th), s * math.sin(th)
        R = np.array([[a, -b], [b, a]])
        tx, ty = cd - R @ c
        self.M = np.float32([[a, -b, tx], [b, a, ty]])
        self._t = now
        return self.M

    def hold(self, now: float) -> np.ndarray | None:
        if self.M is not None and now - self._t <= HOLD_S:
            return self.M
        self.reset()
        return None


def _similarity(mf: MemeFace, face_pts: np.ndarray) -> np.ndarray | None:
    fw = float(np.linalg.norm(face_pts[FACE_RIGHT_EDGE] - face_pts[FACE_LEFT_EDGE]))
    if fw < 8:
        return None
    src = mf.anchors
    if src is not None:
        dst = face_pts[list(ANCHORS)]
        M, _ = cv2.estimateAffinePartial2D(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), method=cv2.RANSAC,
                                           ransacReprojThreshold=max(3.0, 0.06 * fw))
        if M is not None and np.isfinite(M).all() and 0.02 < math.hypot(M[0, 0], M[1, 0]) < 50:
            return M
    return _bbox_similarity(mf.bgra.shape, face_pts)


def _bbox_similarity(shape: tuple[int, ...], face_pts: np.ndarray) -> np.ndarray:
    """Fallback: map the cutout's (alpha-cropped) box onto the face box, rotated with the eye line."""
    h, w = shape[:2]
    (x0, y0), (x1, y1) = face_pts.min(axis=0), face_pts.max(axis=0)
    s = 1.3 * math.sqrt(max(1.0, float((x1 - x0) * (y1 - y0))) / (w * h))
    v = face_pts[FACE_EYE_CORNERS[3]] - face_pts[FACE_EYE_CORNERS[0]]
    th = math.atan2(float(v[1]), float(v[0]))
    a, b = s * math.cos(th), s * math.sin(th)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return np.float32([[a, -b, cx - (a * w / 2 - b * h / 2)], [b, a, cy - (b * w / 2 + a * h / 2)]])


def _cut_hand(alpha: np.ndarray, pts: np.ndarray) -> None:
    """Zero alpha under a hand (convex hull grown ~18%, plus the start of the forearm)."""
    forearm = pts[WRIST] + (pts[WRIST] - pts[MIDDLE_MCP]) * 1.2
    pts = np.vstack([pts, forearm[None]])
    hull = cv2.convexHull(_expand(pts, 1.18).astype(np.int32))
    cv2.fillConvexPoly(alpha, hull, 0)


def _expand(pts: np.ndarray, k: float) -> np.ndarray:
    c = pts.mean(axis=0)
    return c + (pts - c) * k


# ================================================================ tone match
class _ToneMatch:
    """Reinhard LAB mean/std transfer toward the user's face, as a LUT; mostly lightness, a little colour."""

    STRENGTH = np.float32([0.7, 0.35, 0.35])  # L, a, b

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.lut: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._t = -math.inf

    def observe(self, buf: np.ndarray, face_pts: np.ndarray, mf: MemeFace, now: float) -> None:
        if now - self._t < 0.25:
            return
        self._t = now
        (x0, y0), (x1, y1) = face_pts.min(axis=0), face_pts.max(axis=0)
        cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 0.25, (y1 - y0) * 0.25
        H, W = buf.shape[:2]
        rx0, ry0 = max(0, int(cx - hw)), max(0, int(cy - hh))
        rx1, ry1 = min(W, int(cx + hw)), min(H, int(cy + hh))
        if rx1 - rx0 < 4 or ry1 - ry0 < 4:
            return
        small = cv2.resize(buf[ry0:ry1, rx0:rx1], (24, 24), interpolation=cv2.INTER_AREA)
        mean, std = cv2.meanStdDev(cv2.cvtColor(small, cv2.COLOR_BGR2LAB))
        mean, std = mean.ravel().astype(np.float32), std.ravel().astype(np.float32)
        if self._mean is None:
            self._mean, self._std = mean, std
        else:
            self._mean = 0.7 * self._mean + 0.3 * mean
            self._std = 0.7 * self._std + 0.3 * std
        self.lut = _tone_lut(mf.lab_mean, mf.lab_std, self._mean, self._std, self.STRENGTH)


def _tone_lut(src_mean: np.ndarray, src_std: np.ndarray, dst_mean: np.ndarray, dst_std: np.ndarray,
              strength: np.ndarray) -> np.ndarray:
    x = np.arange(256, dtype=np.float32)[:, None]
    ratio = np.clip(dst_std / np.maximum(src_std, 1e-3), 0.5, 2.0)
    y = (x - src_mean) * ratio + dst_mean
    y = x + strength * (y - x)
    return np.clip(y + 0.5, 0, 255).astype(np.uint8).reshape(256, 1, 3)


# ================================================================ tears
class _Tears:
    """Tear particles in fixed-size arrays; spawned at the lower eyelids, stamped in one scatter."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.pos = np.zeros((MAX_TEARS, 2), np.float32)
        self.vel = np.zeros((MAX_TEARS, 2), np.float32)
        self.age = np.zeros(MAX_TEARS, np.float32)
        self.life = np.ones(MAX_TEARS, np.float32)
        self.alive = np.zeros(MAX_TEARS, bool)
        self._acc = np.zeros(2)
        self._size = 12

    def reset(self) -> None:
        self.alive[:] = False
        self._acc = self.rng.uniform(0.5, 1.0, 2)  # first tears come quickly

    @property
    def count(self) -> int:
        return int(self.alive.sum())

    def step(self, dt: float, face_pts: np.ndarray | None, fw: float) -> None:
        a = self.alive
        self.pos[a] += self.vel[a] * dt
        self.age[a] += dt
        self.alive &= self.age < self.life
        if face_pts is None or fw < 20:
            return
        k = min(2.5, max(0.4, fw / 250))
        self._size = max(6, round(fw * 0.055 / 2) * 2)
        self._acc += TEAR_RATE * dt * self.rng.uniform(0.6, 1.4, 2)
        nose_x = face_pts[FACE_NOSE_BRIDGE, 0]
        for eye, lid_idx in enumerate(FACE_LOWER_LIDS):  # two eyes, not per particle
            n = int(self._acc[eye])
            if n <= 0:
                continue
            self._acc[eye] -= n
            free = np.flatnonzero(~self.alive)[:n]
            m = len(free)
            if m == 0:
                continue
            lid = face_pts[lid_idx]
            outward = math.copysign(1.0, lid[0] - nose_x) if lid[0] != nose_x else (-1.0 if eye == 0 else 1.0)
            self.pos[free, 0] = lid[0] + self.rng.normal(0, 0.012 * fw, m)
            self.pos[free, 1] = lid[1] + 0.02 * fw
            self.vel[free, 0] = outward * self.rng.uniform(8, 26, m) * k
            self.vel[free, 1] = TEAR_SPEED * k * self.rng.uniform(0.8, 1.2, m)
            self.age[free] = 0
            self.life[free] = self.rng.uniform(1.0, 1.6, m)
            self.alive[free] = True

    def draw(self, out: np.ndarray) -> None:
        idx = np.flatnonzero(self.alive)
        if len(idx) == 0:
            return
        pm, a, oy, ox, h, w = _tear_sprite(self._size)
        left = self.life[idx] - self.age[idx]
        opacity = np.clip(left / (0.3 * self.life[idx]), 0, 1) * np.clip(self.age[idx] / 0.08, 0, 1)
        xs = (self.pos[idx, 0] - w / 2).astype(np.int32)
        ys = (self.pos[idx, 1] - h / 2).astype(np.int32)
        stamp(out, pm, a, oy, ox, xs, ys, opacity.astype(np.float32))


@lru_cache(maxsize=32)
def _tear_sprite(h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Teardrop as sparse (premultiplied colour, alpha, dy, dx) for the pixels it covers."""
    w = max(3, round(h * 0.62))
    ss = 4
    H2, W2 = h * ss, w * ss
    body = np.zeros((H2, W2), np.uint8)
    cx, cy, rx, ry = W2 / 2, H2 * 0.64, W2 / 2 - 1, H2 * 0.36 - 1
    cv2.ellipse(body, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    tip = np.int32([[cx, 1], [cx - rx * 0.86, cy - ry * 0.5], [cx + rx * 0.86, cy - ry * 0.5]])
    cv2.fillConvexPoly(body, tip, 255, cv2.LINE_AA)
    hi = np.zeros_like(body)
    cv2.ellipse(hi, (int(cx - rx * 0.38), int(cy - ry * 0.25)), (max(1, int(rx * 0.26)), max(1, int(ry * 0.34))),
                0, 0, 360, 255, -1, cv2.LINE_AA)
    body = cv2.resize(body, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    hi = cv2.resize(hi, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 * body
    alpha = body * (TEAR_ALPHA * (1 - hi) + 0.9 * hi)
    colour = np.float32(TEAR_BGR)[None, None, :] * (body * TEAR_ALPHA * (1 - hi))[..., None]
    colour += 255.0 * (0.9 * hi)[..., None]
    oy, ox = np.nonzero(alpha > 0.01)
    return colour[oy, ox], alpha[oy, ox], oy.astype(np.int32), ox.astype(np.int32), h, w


def stamp(out: np.ndarray, pm: np.ndarray, a: np.ndarray, oy: np.ndarray, ox: np.ndarray,
          xs: np.ndarray, ys: np.ndarray, opacity: np.ndarray) -> None:
    """Composite one sparse sprite at many top-left positions with a single gather and scatter.

    Overlapping copies resolve last-writer-wins, which is invisible for small sprites.
    """
    H, W = out.shape[:2]
    if not out.flags.c_contiguous:
        tmp = np.ascontiguousarray(out)
        stamp(tmp, pm, a, oy, ox, xs, ys, opacity)
        np.copyto(out, tmp)
        return
    px = xs[:, None] + ox[None, :]
    py = ys[:, None] + oy[None, :]
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    if not ok.any():
        return
    flat_idx = (py * W + px)[ok]
    aa = (a[None, :] * opacity[:, None])[ok]
    col = (pm[None, :, :] * opacity[:, None, None])[ok]
    flat = out.reshape(-1, 3)
    base = np.take(flat, flat_idx, axis=0).astype(np.float32)
    res = np.clip(base * (1 - aa)[:, None] + col + 0.5, 0, 255).astype(np.uint8)
    flat.view("V3").ravel()[flat_idx] = res.view("V3").ravel()


# ================================================================ cached full-frame assets
def _grade_matrix() -> np.ndarray:
    """3x4 cv2.transform matrix: 70% desaturation, then the cool channel gains, plus a blue lift."""
    luma = np.float32([0.114, 0.587, 0.299])  # BGR
    desat = (1 - DESAT) * np.eye(3, dtype=np.float32) + DESAT * np.tile(luma, (3, 1))
    m = np.diag(np.float32(TINT_BGR)) @ desat
    return np.hstack([m, np.float32(LIFT_BGR)[:, None]]).astype(np.float32)


@lru_cache(maxsize=2)
def _vignette(W: int, H: int) -> np.ndarray:
    v = vignette_map(W, H, strength=0.55, radius=0.95)
    return (v.astype(np.float32) * DARKEN).astype(np.uint8)


@lru_cache(maxsize=2)
def _rain_texture(W: int, H: int) -> np.ndarray:
    """Seamless additive rain: a W x H tile of slanted streaks, wrapped, repeated 2x2 for scrolling."""
    rng = np.random.default_rng(11)
    k = W / 1280
    n = max(50, int(RAIN_STREAKS * W * H / (1280 * 720)))
    ang = math.radians(RAIN_SLANT_DEG)
    length = rng.uniform(15, 35, n) * k
    x0, y0 = rng.uniform(0, W, n), rng.uniform(0, H, n)
    segs = np.stack([np.stack([x0, y0], -1), np.stack([x0 + length * math.sin(ang), y0 + length * math.cos(ang)], -1)],
                    axis=1)
    value = rng.uniform(40, 90, n)
    edges = np.linspace(40, 90, 6)
    bucket = np.clip(np.digitize(value, edges) - 1, 0, 4)
    tile = np.zeros((H, W), np.uint8)
    for b in range(5):  # one polylines call per brightness bucket and wrap offset
        sel = segs[bucket == b]
        v = int(1.6 * (edges[b] + edges[b + 1]) / 2)  # brighter to survive the motion blur below
        for dx, dy in ((0, 0), (-W, 0), (0, -H), (-W, -H)):
            pts = np.round((sel + (dx, dy)) * 4).astype(np.int32)
            cv2.polylines(tile, list(pts), False, v, 1, cv2.LINE_AA, shift=2)
    p = 8
    padded = cv2.copyMakeBorder(tile, p, p, p, p, cv2.BORDER_WRAP)
    tile = cv2.GaussianBlur(padded, (3, 9), sigmaX=0.6, sigmaY=2.0)[p:-p, p:-p]
    tile3 = cv2.merge([tile, (tile * 0.93).astype(np.uint8), (tile * 0.82).astype(np.uint8)])
    return np.ascontiguousarray(np.tile(tile3, (2, 2, 1)))


def _rain_offset(t: float, W: int, H: int) -> tuple[int, int]:
    d = RAIN_SPEED * t
    ang = math.radians(RAIN_SLANT_DEG)
    return int(-d * math.cos(ang)) % H, int(-d * math.sin(ang)) % W


@lru_cache(maxsize=2)
def _caption_sprite(W: int) -> tuple[np.ndarray, np.ndarray]:
    k = W / 1280
    style = TextStyle(font=CAPTION_FONT, size=max(14, int(46 * k)), color=(232, 238, 255, 255),
                      shadow=max(2, int(5 * k)), shadow_alpha=0.8)
    return make_premult(render_text(CAPTION, style))


@lru_cache(maxsize=2)
def _emoji_sprite(W: int) -> tuple[np.ndarray, np.ndarray]:
    return make_premult(render_emoji(LOUDLY_CRYING, max(24, int(140 * W / 1280))))


def _zoom_crop(W: int, H: int, s: float, cx: float, cy: float) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """Integer crop (x0, y0, w, h) for a punch-in by ~s near (cx, cy), plus the frame -> output matrix.

    Resizing an integer crop is ~2.3x cheaper than a full-frame warpAffine; the
    1-pixel crop steps are invisible over a slow 8% push-in.
    """
    cw, ch = min(W, max(1, round(W / s))), min(H, max(1, round(H / s)))
    x0 = int(min(max(cx - cw / 2, 0.0), W - cw))
    y0 = int(min(max(cy - ch / 2, 0.0), H - ch))
    sx, sy = W / cw, H / ch
    return (x0, y0, cw, ch), np.float32([[sx, 0, -x0 * sx], [0, sy, -y0 * sy]])


def _blend_u8(roi: np.ndarray, bgr: np.ndarray, alpha: np.ndarray) -> None:
    """roi = bgr*a + roi*(1-a) in place with uint8 alpha (2x cheaper than float blendLinear here)."""
    a3 = cv2.merge([alpha, alpha, alpha])
    cv2.multiply(bgr, a3, dst=bgr, scale=1 / 255)
    cv2.bitwise_not(a3, dst=a3)
    cv2.multiply(roi, a3, dst=roi, scale=1 / 255)
    cv2.add(roi, bgr, dst=roi)


def _apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (np.asarray(pts, np.float32) @ M[:, :2].T + M[:, 2]).astype(np.float32)
