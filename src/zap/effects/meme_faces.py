"""Face landmarks in meme pictures and clip frames, found off the render thread.

Cutaways use them to line the meme's face up with the user's for the morph
transition, and to put the user's face onto the meme (face swap). Detection
runs once per asset and output size on a background thread, at most ~40 frames
per clip (the rest are filled from the nearest detected frame), and is cached
on disk so later runs start instantly.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading

import cv2
import numpy as np

from zap import paths
from zap.fetch_models import model_path

log = logging.getLogger(__name__)

MAX_DETECT = 120  # frames actually run through the face model per clip (once; cached on disk)
CACHE_VERSION = 4

_LOCK = threading.Lock()
_DETECTOR = None


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        os.environ.setdefault("GLOG_minloglevel", "2")
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_path("face_landmarker.task").read_bytes()),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
        )
        _DETECTOR = vision.FaceLandmarker.create_from_options(opts)
    return _DETECTOR


def detect(bgr: np.ndarray, near: np.ndarray | None = None) -> np.ndarray | None:
    """(478, 2) float32 face landmarks in ``bgr``'s pixels, or None (no face, or no model).

    The face model wants faces that fill a good part of the picture, so small faces (an actor in a
    wide shot) are looked for in zoomed crops: around ``near`` (last frame's landmarks) first, then
    over a grid of overlapping crops.
    """
    if not model_path("face_landmarker.task").is_file():
        return None
    h, w = bgr.shape[:2]
    if near is not None:
        c = near.mean(axis=0)
        r = 1.6 * max(float(np.ptp(near[:, 0])), float(np.ptp(near[:, 1])), 24.0)
        found = _detect_crop(bgr, int(c[0] - r), int(c[1] - r), int(2 * r), int(2 * r))
        if found is not None:
            return found
    found = _detect_whole(bgr)
    if found is not None:
        return found
    for cw, ch in ((w // 2, h // 2), (w // 3, h // 3)):
        for y in range(0, h - ch + 1, max(1, ch // 2)):
            for x in range(0, w - cw + 1, max(1, cw // 2)):
                found = _detect_crop(bgr, x, y, cw, ch)
                if found is not None:
                    return found
    return None


def _detect_crop(bgr: np.ndarray, x: int, y: int, cw: int, ch: int) -> np.ndarray | None:
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + cw), min(h, y + ch)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    crop = bgr[y0:y1, x0:x1]
    k = 512 / max(crop.shape[:2])  # the model sees the crop at a useful size
    big = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
    found = _detect_whole(big)
    return None if found is None else (found / np.float32(k) + np.float32([x0, y0])).astype(np.float32)


def _detect_whole(bgr: np.ndarray) -> np.ndarray | None:
    import mediapipe as mp

    with _LOCK:
        det = _detector()
        for pad_frac in (0.0, 0.25):  # padding helps when the face fills the picture
            pad = int(pad_frac * max(bgr.shape[:2]))
            img = bgr if pad == 0 else cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                                                          value=(128, 128, 128))
            rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if res.face_landmarks:
                pts = np.array([[q.x, q.y] for q in res.face_landmarks[0]], np.float32)
                return pts * np.float32([img.shape[1], img.shape[0]]) - np.float32(pad)
    return None


def _key(frames: list[np.ndarray], tag: str) -> str:
    h = hashlib.sha1(tag.encode())
    for f in (frames[0], frames[len(frames) // 2], frames[-1]):
        h.update(cv2.resize(f, (32, 18), interpolation=cv2.INTER_AREA).tobytes())
    h.update(f"{len(frames)}x{frames[0].shape}".encode())
    return h.hexdigest()[:16]


def _cache_path(key: str):
    return paths.calibration_dir() / f"meme_frames_{key}.npz"


def _fill(pts: list[np.ndarray | None]) -> list[np.ndarray | None]:
    """Fill frames without landmarks: interpolated between the frames around them, else the nearest."""
    found = [i for i, p in enumerate(pts) if p is not None]
    if not found:
        return pts
    out: list[np.ndarray | None] = []
    for i, p in enumerate(pts):
        prev = max((j for j in found if j <= i), default=None)
        nxt = min((j for j in found if j >= i), default=None)
        if p is not None or prev is None or nxt is None:
            out.append(p if p is not None else pts[prev if nxt is None else nxt])
        else:
            w = (i - prev) / (nxt - prev)
            out.append(((1 - w) * pts[prev] + w * pts[nxt]).astype(np.float32))
    return out


class FrameFaces:
    """Landmarks for each frame of a meme, filled in by a background thread."""

    def __init__(self, frames: list[np.ndarray], tag: str = "") -> None:
        self.n = len(frames)
        self.pts: list[np.ndarray | None] = [None] * self.n
        self.found: list[bool] = [False] * self.n  # the face model really ran on this frame and saw a face
        self.ready = threading.Event()
        if self.n:
            threading.Thread(target=self._run, args=(frames, _key(frames, tag)), name="zap-meme-faces",
                             daemon=True).start()
        else:
            self.ready.set()

    def _run(self, frames: list[np.ndarray], key: str) -> None:
        try:
            p = _cache_path(key)
            if p.is_file():
                data = np.load(p)
                if int(data["version"]) == CACHE_VERSION and len(data["valid"]) == self.n:
                    self.pts = [data["pts"][i] if data["valid"][i] else None for i in range(self.n)]
                    self.found = [bool(f) for f in data["found"]]
                    return
            step = max(1, -(-self.n // MAX_DETECT))
            pts: list[np.ndarray | None] = [None] * self.n
            last = None
            for i in range(0, self.n, step):
                pts[i] = detect(frames[i], last)
                last = pts[i] if pts[i] is not None else last
            self.found = [q is not None for q in pts]
            self.pts = _fill(pts)
            valid = np.array([q is not None for q in self.pts])
            arr = np.stack([q if q is not None else np.zeros((478, 2), np.float32) for q in self.pts])
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, version=CACHE_VERSION, valid=valid, pts=arr, found=np.array(self.found))
            if not valid.any():
                log.info("no face found in a meme; it will cross-fade instead of morphing")
        except Exception:
            log.exception("finding meme face landmarks failed; cross-fading instead")
        finally:
            self.ready.set()

    def get(self, i: int) -> np.ndarray | None:
        return self.pts[i % self.n] if self.n else None
