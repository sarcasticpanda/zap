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

MAX_DETECT = 40  # frames actually run through the face model per clip
CACHE_VERSION = 2

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


def detect(bgr: np.ndarray) -> np.ndarray | None:
    """(478, 2) float32 face landmarks in ``bgr``'s pixels, or None (no face, or no model)."""
    if not model_path("face_landmarker.task").is_file():
        return None
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
    """Give every frame the landmarks of the nearest frame that has some."""
    found = [i for i, p in enumerate(pts) if p is not None]
    if not found:
        return pts
    return [pts[min(found, key=lambda j: abs(j - i))] for i in range(len(pts))]


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
            for i in range(0, self.n, step):
                pts[i] = detect(frames[i])
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
