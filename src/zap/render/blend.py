"""Fast compositing helpers.

Rules that keep a frame under budget at 1280x720 (measured on the target
laptop): no full-frame numpy float math, blend only inside the region a sprite
covers, prefer cv2 ops with preallocated ``dst`` buffers, precompute LUTs and
maps once.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


# ---------------------------------------------------------------- easing
def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else min(x, 1.0)


def ease_out_cubic(x: float) -> float:
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_in_quint(x: float) -> float:
    return clamp01(x) ** 5


def ease_out_back(x: float, overshoot: float = 1.70158) -> float:
    x = clamp01(x)
    c3 = overshoot + 1
    return 1 + c3 * (x - 1) ** 3 + overshoot * (x - 1) ** 2


def ease_in_out_sine(x: float) -> float:
    return -(math.cos(math.pi * clamp01(x)) - 1) / 2


# ---------------------------------------------------------------- sprites
def make_premult(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RGBA (Pillow order) -> (premultiplied BGR, 3-channel inverse alpha), both uint8."""
    bgr = cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]), cv2.COLOR_RGB2BGR)
    a = np.ascontiguousarray(rgba[..., 3])
    a3 = cv2.merge([a, a, a])
    pm = cv2.multiply(bgr, a3, scale=1 / 255)
    inv3 = cv2.subtract(np.full_like(a3, 255), a3)
    return pm, inv3


def blit_premult(dst: np.ndarray, pm: np.ndarray, inv3: np.ndarray, x: int, y: int) -> None:
    """Composite a premultiplied sprite with its top-left at (x, y); clipped to dst."""
    H, W = dst.shape[:2]
    h, w = pm.shape[:2]
    dx0, dy0 = max(0, x), max(0, y)
    dx1, dy1 = min(W, x + w), min(H, y + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    sx0, sy0 = dx0 - x, dy0 - y
    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)
    roi = dst[dy0:dy1, dx0:dx1]
    cv2.multiply(roi, inv3[sy0:sy1, sx0:sx1], dst=roi, scale=1 / 255)
    cv2.add(roi, pm[sy0:sy1, sx0:sx1], dst=roi)


def blit_premult_centered(dst: np.ndarray, pm: np.ndarray, inv3: np.ndarray, cx: float, cy: float) -> None:
    h, w = pm.shape[:2]
    blit_premult(dst, pm, inv3, round(cx - w / 2), round(cy - h / 2))


def blit_rgba(dst: np.ndarray, rgba: np.ndarray, x: int, y: int, opacity: float = 1.0) -> None:
    """Convenience: composite an RGBA sprite (not cached as premultiplied)."""
    if opacity < 1.0:
        rgba = rgba.copy()
        rgba[..., 3] = (rgba[..., 3].astype(np.float32) * clamp01(opacity)).astype(np.uint8)
    pm, inv3 = make_premult(rgba)
    blit_premult(dst, pm, inv3, x, y)


def blend_masked(dst: np.ndarray, src: np.ndarray, alpha: np.ndarray) -> None:
    """dst = src*alpha + dst*(1-alpha) in place. alpha is float32 HxW in [0, 1]."""
    cv2.blendLinear(src, dst, alpha, 1.0 - alpha, dst=dst)


# ---------------------------------------------------------------- grading
def s_curve_lut(softness: float = 28.0) -> np.ndarray:
    v = np.arange(256, dtype=np.float32)
    lut = 255.0 / (1.0 + np.exp(-(v - 128.0) / softness))
    lut = (lut - lut.min()) / (lut.max() - lut.min()) * 255.0
    return lut.astype(np.uint8)


def channel_luts(b: float = 1.0, g: float = 1.0, r: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    """A (256, 1, 3) LUT that scales each channel, usable with cv2.LUT on BGR."""
    v = (np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0
    lut = np.stack([v * b, v * g, v * r], axis=-1).clip(0, 255).astype(np.uint8)
    return lut.reshape(256, 1, 3)


def vignette_map(w: int, h: int, strength: float = 0.5, radius: float = 0.85) -> np.ndarray:
    """uint8 3-channel multiplier map (255 = untouched) for cv2.multiply(scale=1/255)."""
    xs = np.linspace(-1, 1, w, dtype=np.float32)
    ys = np.linspace(-1, 1, h, dtype=np.float32)
    d = np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2) / math.sqrt(2) / radius
    m = 1.0 - strength * np.clip(d, 0, 1) ** 2
    m8 = (m * 255).astype(np.uint8)
    return cv2.merge([m8, m8, m8])


def grain_frames(w: int, h: int, sigma: float = 12.0, count: int = 6, seed: int = 7) -> list[np.ndarray]:
    """Half-resolution int16 noise frames upsampled with nearest neighbour (coarse grain survives Zoom)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        n = rng.normal(0, sigma, (h // 2, w // 2)).astype(np.int16)
        out.append(cv2.resize(n, (w, h), interpolation=cv2.INTER_NEAREST))
    return out


def add_grain_gray(gray: np.ndarray, noise: np.ndarray) -> np.ndarray:
    return np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- geometry
def zoom_about(frame: np.ndarray, scale: float, cx: float, cy: float, out: np.ndarray | None = None) -> np.ndarray:
    """Punch-in by `scale` (>=1) centred near (cx, cy), keeping the crop inside the frame."""
    H, W = frame.shape[:2]
    if scale <= 1.001:
        if out is None:
            return frame.copy()
        np.copyto(out, frame)
        return out
    cw, ch = W / scale, H / scale
    x0 = min(max(cx - cw / 2, 0), W - cw)
    y0 = min(max(cy - ch / 2, 0), H - ch)
    M = np.float32([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]])
    return cv2.warpAffine(frame, M, (W, H), dst=out, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def shake_offset(t: float, amplitude: float, decay: float = 6.0, freq: float = 31.0) -> tuple[int, int]:
    a = amplitude * math.exp(-decay * t)
    return int(a * math.sin(freq * t)), int(a * math.cos(freq * 1.3 * t + 1.1))


def translate(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return frame
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    H, W = frame.shape[:2]
    return cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REPLICATE)


def cover_fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Scale and centre-crop an image so it fills w x h (like CSS object-fit: cover)."""
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    nw, nh = max(w, round(iw * s)), max(h, round(ih * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return np.ascontiguousarray(resized[y0:y0 + h, x0:x0 + w])


def upsample_mask(mask: np.ndarray, w: int, h: int, blur: int = 0) -> np.ndarray:
    """Float mask at inference resolution -> float32 at output size; optional blur at low res first."""
    m = mask
    if blur:
        k = blur | 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
