"""User-supplied meme images, clips and sounds.

Meme pictures and sounds are copyrighted, so ZAP never ships them. Users drop
their own files into the ZAP home folder (``assets/memes`` and ``assets/sfx``),
named by key, e.g. ``jim_stare.png`` or ``money.wav``. Every lookup that misses
returns None and the effect falls back to a procedural version; the miss is
remembered so the HUD can tell the user which file to add.

GIFs are decoded with Pillow (OpenCV's FFmpeg build can't decode some of them,
"Could not find decoder for codec_id=61"); its per-frame delays are honoured by
resampling the frames onto an even timeline. Videos go through OpenCV. Clip
frames are shrunk to fit the 1280x720 output so a 4 s 1080p clip stays small.
"""

from __future__ import annotations

import logging
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from zap import paths

log = logging.getLogger(__name__)

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
CLIP_EXT = (".mp4", ".gif", ".mov", ".webm", ".avi")
SOUND_EXT = (".wav",)
SAMPLE_RATE = 48000
MAX_CLIP_SECONDS = 4.0
MAX_CLIP_SIZE = (1280, 720)  # clip frames are shrunk (never grown) to fit this box
GIF_DEFAULT_DELAY_MS = 100  # browsers play 0-10 ms GIF delays at 100 ms; so do we
MAX_CLIP_FPS = 60.0


@dataclass
class Clip:
    frames: list[np.ndarray]  # BGR uint8
    fps: float

    @property
    def duration(self) -> float:
        return len(self.frames) / self.fps


class AssetResolver:
    def __init__(self, memes: Path | None = None, sfx: Path | None = None) -> None:
        self.memes = memes or paths.memes_dir()
        self.sfx = sfx or paths.sfx_dir()
        self._cache: dict[tuple[str, str], object] = {}
        self.missing: set[str] = set()

    def _find(self, folder: Path, key: str, exts: tuple[str, ...]) -> Path | None:
        for ext in exts:
            p = folder / f"{key}{ext}"
            if p.is_file():
                return p
        return None

    def _miss(self, what: str) -> None:
        if what not in self.missing:
            self.missing.add(what)
            log.info("asset not found: %s (using fallback)", what)

    def image(self, key: str) -> np.ndarray | None:
        """BGRA uint8 (alpha = 255 when the file has none), or None."""
        ck = ("image", key)
        if ck not in self._cache:
            p = self._find(self.memes, key, IMAGE_EXT)
            img = None
            if p is not None:
                # imdecode handles non-ASCII paths, unlike imread on Windows
                img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if img is None:
                    img = _pil_image(p)
                elif img.dtype != np.uint8:  # 16-bit PNG
                    img = (img // 257).astype(np.uint8)
                if img is not None:
                    if img.ndim == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                    elif img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            if img is None:
                self._miss(f"assets/memes/{key}.png")
            self._cache[ck] = img
        return self._cache[ck]  # type: ignore[return-value]

    def clip(self, key: str) -> Clip | None:
        """A short video/gif as BGR frames (first MAX_CLIP_SECONDS, fitted into MAX_CLIP_SIZE), or None."""
        ck = ("clip", key)
        if ck not in self._cache:
            p = self._find(self.memes, key, CLIP_EXT)
            clip = None
            if p is not None:
                if p.suffix.lower() == ".gif":
                    clip = read_gif(p)
                    if clip is None:
                        clip = read_video(p)  # not a GIF Pillow can read; maybe FFmpeg can
                else:
                    clip = read_video(p)
                if clip is None:
                    log.warning("could not decode %s", p.name)
            self._cache[ck] = clip
        return self._cache[ck]  # type: ignore[return-value]

    def sound(self, key: str) -> np.ndarray | None:
        """Mono float32 at 48 kHz, or None. Only WAV is read (stdlib, no extra dependency)."""
        ck = ("sound", key)
        if ck not in self._cache:
            p = self._find(self.sfx, key, SOUND_EXT)
            data = None
            if p is not None:
                try:
                    data = read_wav(p)
                except (wave.Error, ValueError, EOFError) as e:
                    log.warning("could not read %s: %s", p.name, e)
            self._cache[ck] = data
        return self._cache[ck]  # type: ignore[return-value]


def _shrink(bgr: np.ndarray, box: tuple[int, int] = MAX_CLIP_SIZE) -> np.ndarray:
    """Scale a frame down (never up) so it fits inside box (w, h)."""
    h, w = bgr.shape[:2]
    s = min(box[0] / w, box[1] / h)
    if s >= 1.0:
        return np.ascontiguousarray(bgr)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    return cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)


def _pil_image(path: Path) -> np.ndarray | None:
    """BGRA via Pillow, for files OpenCV can't decode (e.g. a .png that is really a GIF)."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            rgba = np.asarray(im.convert("RGBA"))
    except (OSError, ValueError) as e:
        log.warning("could not read %s: %s", path.name, e)
        return None
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)


def read_gif(path: Path, max_seconds: float = MAX_CLIP_SECONDS) -> Clip | None:
    """Decode an animated (or still) GIF with Pillow into an even-fps Clip.

    Pillow applies each frame's disposal, so every frame comes out fully
    composited; transparent pixels end up black. Frames with equal delays map
    1:1 onto the clip; mixed delays (30/40/70 ms) are resampled onto a grid at
    the shortest delay, repeating frame references (no copies) so held frames
    stay held.
    """
    from PIL import Image, ImageSequence

    frames: list[np.ndarray] = []
    delays: list[int] = []
    try:
        with Image.open(path) as im:
            total = 0
            for fr in ImageSequence.Iterator(im):
                d = fr.info.get("duration") or 0
                d = int(d) if d > 10 else GIF_DEFAULT_DELAY_MS
                rgba = np.asarray(fr.convert("RGBA"))
                a = rgba[..., 3:4].astype(np.uint16)
                rgb = (rgba[..., :3] * a // 255).astype(np.uint8) if a.min() < 255 else rgba[..., :3]
                frames.append(_shrink(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)))
                delays.append(d)
                total += d
                if total >= max_seconds * 1000:
                    break
    except (OSError, ValueError, EOFError) as e:
        if not frames:
            log.warning("could not read %s with Pillow: %s", path.name, e)
            return None
        log.warning("%s is truncated (%s); using its first %d frames", path.name, e, len(frames))
    if not frames:
        return None
    if len(set(delays)) == 1:
        fps = min(MAX_CLIP_FPS, 1000.0 / delays[0])
        n = min(len(frames), max(1, round(max_seconds * fps)))
        return Clip(frames[:n], fps)
    step = max(1000.0 / MAX_CLIP_FPS, float(min(delays)))
    ends = np.cumsum(delays)
    total = min(float(ends[-1]), max_seconds * 1000)
    n = max(1, math.ceil(total / step - 1e-6))
    idx = np.searchsorted(ends, (np.arange(n) + 0.5) * step, side="right")
    return Clip([frames[min(int(i), len(frames) - 1)] for i in idx], 1000.0 / step)


def read_video(path: Path, max_seconds: float = MAX_CLIP_SECONDS) -> Clip | None:
    """Decode the first max_seconds of a video with OpenCV (FFmpeg)."""
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fps = fps if 1.0 <= fps <= 120.0 else 25.0
        frames = []
        while len(frames) < max(1, int(fps * max_seconds)):
            ok, f = cap.read()
            if not ok:
                break
            frames.append(_shrink(f))
    finally:
        cap.release()
    return Clip(frames, fps) if frames else None


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    elif width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        x = ((b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)) << 8 >> 8)
        x = x.astype(np.float32) / 8388608
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648
    else:
        raise ValueError(f"unsupported sample width {width}")
    x = x.reshape(-1, ch).mean(axis=1)
    if rate != SAMPLE_RATE:
        t_old = np.arange(len(x)) / rate
        t_new = np.arange(int(len(x) * SAMPLE_RATE / rate)) / SAMPLE_RATE
        x = np.interp(t_new, t_old, x).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)
