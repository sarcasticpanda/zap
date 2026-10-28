"""MediaPipe perception: one camera frame in, hands + face + person mask out.

Three MediaPipe Tasks (gesture recognizer, face landmarker, selfie segmenter)
run side by side on a small thread pool, all on one downscaled RGB copy of the
frame; their C++ calls go through ctypes, which releases the GIL, so they really
overlap. Hands run every frame while a hand is visible because snaps and poses
are fast; with no hand in view the recognizer runs palm detection on every call
(~25 ms, the most expensive model), so it then only runs every
``hands_idle_every`` frames. The face and the mask move slowly and run on a
cadence unless an active effect needs them fresh. Every landmark leaves this module in OUTPUT pixel coordinates so nothing
downstream needs to know the inference resolution.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Self

import cv2
import numpy as np

from zap.config import PerceptionCfg
from zap.fetch_models import model_path
from zap.signals import Face, FramePacket, Hand, Signals

# Must be set before mediapipe is imported (it is imported lazily in load_tasks).
os.environ.setdefault("GLOG_minloglevel", "2")

log = logging.getLogger(__name__)

FACE_CACHE_S = 1.0  # reuse the last face this long when the model skips a frame or loses it
TASK_KEYS = ("hands", "face", "mask")
TIMING_KEYS = ("prep", *TASK_KEYS, "total")

GESTURE_MODEL = "gesture_recognizer.task"
FACE_MODEL = "face_landmarker.task"
SEGMENTER_MODEL = "selfie_segmenter_landscape.tflite"


# ---------------------------------------------------------------- pure conversion
def _points(landmarks: Any, out_w: float, out_h: float) -> np.ndarray:
    """Normalized landmarks -> (N, 3) float32 output pixels (z scaled like x)."""
    a = np.array([(p.x or 0.0, p.y or 0.0, p.z or 0.0) for p in landmarks], dtype=np.float32).reshape(-1, 3)
    a *= np.array((out_w, out_h, out_w), dtype=np.float32)
    return a


def _top(categories: Any) -> tuple[str, float]:
    if not categories:
        return "None", 0.0
    best = max(categories, key=lambda c: c.score or 0.0)
    return best.category_name or "None", float(best.score or 0.0)


def convert_hands(result: Any, out_w: int, out_h: int) -> list[Hand]:
    """GestureRecognizerResult -> hands in output pixels, in MediaPipe's order."""
    if result is None:
        return []
    landmarks = getattr(result, "hand_landmarks", None) or []
    handedness = getattr(result, "handedness", None) or []
    gestures = getattr(result, "gestures", None) or []
    hands = []
    for i, lms in enumerate(landmarks):
        if len(lms) != 21:
            continue
        _, score = _top(handedness[i] if i < len(handedness) else None)
        name, gscore = _top(gestures[i] if i < len(gestures) else None)
        hands.append(Hand(lm=_points(lms, out_w, out_h), score=score, gesture=name, gesture_score=gscore))
    return hands


def convert_face(result: Any, out_w: int, out_h: int, t: float = 0.0) -> Face | None:
    """FaceLandmarkerResult -> the first face in output pixels (fresh), or None."""
    faces = getattr(result, "face_landmarks", None) if result is not None else None
    if not faces or not len(faces[0]):
        return None
    blendshapes = getattr(result, "face_blendshapes", None) or []
    blend = {c.category_name: float(c.score) for c in blendshapes[0]} if len(blendshapes) else {}
    mats = getattr(result, "facial_transformation_matrixes", None)
    matrix = None
    if mats is not None and len(mats):
        m = np.array(mats[0], dtype=np.float64)
        matrix = m.reshape(4, 4) if m.size == 16 else None
    return Face(lm=_points(faces[0], out_w, out_h), blend=blend, matrix=matrix, t=t, fresh=True)


def convert_mask(result: Any) -> np.ndarray | None:
    """ImageSegmenterResult -> an owned (H, W) float32 copy of the first confidence mask."""
    masks = getattr(result, "confidence_masks", None) if result is not None else None
    if not masks:
        return None
    view = masks[0].numpy_view() if hasattr(masks[0], "numpy_view") else masks[0]
    m = np.array(view, dtype=np.float32, copy=True)  # the view dies with the result
    if m.ndim == 3 and m.shape[2] == 1:
        m = m.reshape(m.shape[0], m.shape[1])
    if m.ndim != 2:
        raise ValueError(f"unexpected segmenter mask shape {m.shape}")
    return m


# ---------------------------------------------------------------- small state holders
class FaceCache:
    """Bridges frames where the face model skipped or blinked out, for at most ``max_age`` seconds."""

    def __init__(self, max_age: float = FACE_CACHE_S) -> None:
        self.max_age = max_age
        self.last: Face | None = None
        self._stale: Face | None = None

    def update(self, face: Face | None, t: float) -> Face | None:
        if face is not None:
            self.last, self._stale = face, None
            return face
        if self.last is None:
            return None
        if 0.0 <= t - self.last.t <= self.max_age:
            if self._stale is None:
                self._stale = replace(self.last, fresh=False)
            return self._stale
        self.clear()
        return None

    def clear(self) -> None:
        self.last = self._stale = None


class MaskPolarity:
    """Decides once whether the segmenter mask is person or background confidence.

    The value at the face centre settles it: a person is there. Ambiguous samples
    (near 0.5) wait for another frame; after ``max_samples`` the mean decides.
    Until decided the mask is passed through as person confidence, which is what
    the selfie segmenter reports for a frame without anyone in it (all ~0).
    """

    def __init__(self, margin: float = 0.2, max_samples: int = 15) -> None:
        self.margin = margin
        self.max_samples = max_samples
        self.person: bool | None = None
        self._samples: list[float] = []

    @property
    def label(self) -> str:
        return "undecided" if self.person is None else "person" if self.person else "background"

    def observe(self, mask: np.ndarray, face_xy: np.ndarray, out_w: int, out_h: int) -> None:
        if self.person is not None:
            return
        h, w = mask.shape
        x = round(float(face_xy[0]) * w / out_w)
        y = round(float(face_xy[1]) * h / out_h)
        if not (0 <= x < w and 0 <= y < h):
            return
        r = max(1, w // 160)
        v = float(np.median(mask[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1]))
        self._samples.append(v)
        if abs(v - 0.5) > self.margin:
            decided_by = v
        elif len(self._samples) >= self.max_samples:
            decided_by = float(np.mean(self._samples))
        else:
            return
        self.person = decided_by > 0.5
        log.info(
            "segmenter mask is %s confidence (%.2f at the face centre)%s",
            self.label,
            decided_by,
            "" if self.person else "; inverting it",
        )

    def apply(self, mask: np.ndarray) -> np.ndarray:
        if self.person is False:
            np.subtract(1.0, mask, out=mask)
        return mask


class Timings:
    """Last value and a rolling window per key, in milliseconds."""

    def __init__(self, keys: tuple[str, ...] = TIMING_KEYS, window: int = 120) -> None:
        self.last: dict[str, float] = {k: 0.0 for k in keys}
        self._hist: dict[str, deque[float]] = {k: deque(maxlen=window) for k in keys}

    def add(self, key: str, ms: float) -> None:
        self.last[key] = ms
        self._hist.setdefault(key, deque(maxlen=120)).append(ms)

    def percentile(self, key: str, q: float) -> float:
        h = self._hist.get(key)
        return float(np.percentile(np.fromiter(h, dtype=np.float64), q)) if h else 0.0

    def p95(self, key: str) -> float:
        return self.percentile(key, 95)


# ---------------------------------------------------------------- MediaPipe tasks
@dataclass
class TaskSet:
    """The three running tasks plus how to wrap an RGB array for them (fakes in tests)."""

    hands: Any  # .recognize_for_video(image, timestamp_ms)
    face: Any  # .detect_for_video(image, timestamp_ms)
    seg: Any  # .segment_for_video(image, timestamp_ms)
    wrap: Callable[[np.ndarray], Any]

    def close(self) -> None:
        for task in (self.hands, self.face, self.seg):
            try:
                task.close()
            except Exception:  # closing must not hide the real shutdown path
                log.debug("closing %r failed", task, exc_info=True)


def _model_bytes(name: str) -> bytes:
    p = model_path(name)
    if not p.is_file():
        raise FileNotFoundError(f"MediaPipe model missing: {p} (download it with zap.fetch_models.fetch())")
    # A buffer instead of model_asset_path: MediaPipe's own file loader fails on some Windows paths.
    return p.read_bytes()


def load_tasks() -> TaskSet:
    """Build the three tasks in VIDEO mode from the downloaded models."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    video = vision.RunningMode.VIDEO
    built: list[Any] = []
    try:
        hands = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_buffer=_model_bytes(GESTURE_MODEL)),
                running_mode=video,
                num_hands=2,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.5,
            )
        )
        built.append(hands)
        face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=_model_bytes(FACE_MODEL)),
                running_mode=video,
                num_faces=1,
                min_face_detection_confidence=0.6,
                min_face_presence_confidence=0.6,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
        )
        built.append(face)
        seg = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_buffer=_model_bytes(SEGMENTER_MODEL)),
                running_mode=video,
                output_confidence_masks=True,
            )
        )
    except Exception:
        for task in built:
            task.close()
        raise

    def wrap(rgb: np.ndarray) -> Any:
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    return TaskSet(hands=hands, face=face, seg=seg, wrap=wrap)


# ---------------------------------------------------------------- Perception
class Perception:
    """Runs the models on each frame and returns :class:`Signals`.

    Call ``run`` from one thread only (the frame loop). ``pooled=False`` runs the
    three models one after another on the calling thread (used by the bench).
    """

    def __init__(
        self,
        cfg: PerceptionCfg,
        out_w: int,
        out_h: int,
        *,
        pooled: bool = True,
        warmup: bool = True,
        tasks: TaskSet | None = None,
    ) -> None:
        self.cfg = cfg
        self.out_w, self.out_h = int(out_w), int(out_h)
        self.in_w, self.in_h = int(cfg.infer_width), int(cfg.infer_height)
        self.face_every = max(1, int(cfg.face_every))
        self.mask_every = max(1, int(cfg.mask_every))
        self.hands_idle_every = max(1, int(getattr(cfg, "hands_idle_every", 1)))
        self.pooled = pooled
        self._tasks = tasks if tasks is not None else load_tasks()
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="zap-perception")
        self._small = np.empty((self.in_h, self.in_w, 3), np.uint8)
        self._rgb = np.empty((self.in_h, self.in_w, 3), np.uint8)
        self._last_ts = {k: -1 for k in TASK_KEYS}
        self._since = {"hands": 1 << 30, "face": 1 << 30, "mask": 1 << 30}
        self._hands_seen = True  # the last hand run found a hand (start as if it had)
        self._faces = FaceCache()
        self._polarity = MaskPolarity()
        self._mask: np.ndarray | None = None
        self._failed: set[str] = set()
        self.errors = {k: 0 for k in TASK_KEYS}
        self.stats = Timings()
        self.ran: tuple[str, ...] = ()  # which models ran on the last frame
        self.frames = 0
        self._closed = False
        if warmup:
            self._warmup()

    # -------------------------------------------------------- public
    @property
    def timings(self) -> dict[str, float]:
        """Last measured milliseconds for 'prep', 'hands', 'face', 'mask', 'total'."""
        return self.stats.last

    def p95(self, key: str) -> float:
        return self.stats.p95(key)

    @property
    def mask_polarity(self) -> str:
        """'person', 'background' (being inverted) or 'undecided'."""
        return self._polarity.label

    def run(
        self, pkt: FramePacket, need_face: bool = False, need_mask: bool = False, need_hands: bool = False
    ) -> Signals:
        """Signals for one frame. ``need_*`` force a model to run on this frame regardless of cadence."""
        if self._closed:
            raise RuntimeError("Perception is closed")
        t0 = time.perf_counter()
        cv2.resize(pkt.frame, (self.in_w, self.in_h), dst=self._small, interpolation=cv2.INTER_AREA)
        cv2.cvtColor(self._small, cv2.COLOR_BGR2RGB, dst=self._rgb)
        image = self._tasks.wrap(self._rgb)
        t_prep = time.perf_counter()

        jobs = self._schedule(need_face, need_mask, need_hands)
        calls = [(key, self._ts(key, pkt.t)) for key in jobs]
        if self.pooled:
            futures = [(key, self._pool.submit(self._call, key, image, ts, pkt.t)) for key, ts in calls]
            results = {key: f.result() for key, f in futures}
        else:
            results = {key: self._call(key, image, ts, pkt.t) for key, ts in calls}
        del image

        hands: list[Hand] = []
        face_now: Face | None = None
        mask_fresh = False
        for key, (value, ms) in results.items():  # jobs order: hands, face, mask
            self.stats.add(key, ms)
            if key == "hands":
                hands = value or []
                self._hands_seen = bool(hands)
            elif key == "face":
                face_now = value
            elif key == "mask" and value is not None:
                if face_now is not None:  # same image, so the face centre is a fair probe
                    self._polarity.observe(value, face_now.center, self.out_w, self.out_h)
                self._mask = self._polarity.apply(value)
                mask_fresh = True
        face = self._faces.update(face_now, pkt.t)

        self.ran = tuple(jobs)
        self.frames += 1
        t1 = time.perf_counter()
        self.stats.add("prep", (t_prep - t0) * 1e3)
        self.stats.add("total", (t1 - t0) * 1e3)
        return Signals(
            t=pkt.t,
            seq=pkt.seq,
            hands=hands,
            face=face,
            mask=self._mask,
            mask_fresh=mask_fresh,
            width=self.out_w,
            height=self.out_h,
        )

    def reset(self) -> None:
        """Forget the cached face and mask (e.g. after switching cameras)."""
        self._faces.clear()
        self._mask = None
        self._since = {"hands": 1 << 30, "face": 1 << 30, "mask": 1 << 30}
        self._hands_seen = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pool.shutdown(wait=True)
        self._tasks.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -------------------------------------------------------- internals
    def _schedule(self, need_face: bool, need_mask: bool, need_hands: bool = False) -> list[str]:
        jobs = []
        hands_every = 1 if self._hands_seen else self.hands_idle_every
        for key, every, needed in (
            ("hands", hands_every, need_hands),
            ("face", self.face_every, need_face),
            ("mask", self.mask_every, need_mask),
        ):
            self._since[key] += 1
            if needed or self._since[key] >= every:
                self._since[key] = 0
                jobs.append(key)
        return jobs

    def _ts(self, key: str, t: float) -> int:
        """Per-task strictly increasing millisecond timestamps (VIDEO mode requires it)."""
        ms = max(int(t * 1000.0), self._last_ts[key] + 1)
        self._last_ts[key] = ms
        return ms

    def _call(self, key: str, image: Any, ts: int, t: float) -> tuple[Any, float]:
        """Run one model and convert its result; never raises. Returns (value, ms)."""
        t0 = time.perf_counter()
        try:
            if key == "hands":
                value: Any = convert_hands(self._tasks.hands.recognize_for_video(image, ts), self.out_w, self.out_h)
            elif key == "face":
                value = convert_face(self._tasks.face.detect_for_video(image, ts), self.out_w, self.out_h, t)
            else:
                value = convert_mask(self._tasks.seg.segment_for_video(image, ts))
        except Exception:  # one bad frame must not stop the video
            self.errors[key] += 1
            if key not in self._failed:
                self._failed.add(key)
                log.warning("%s model failed; skipping it for this frame (logged once)", key, exc_info=True)
            value = None
        return value, (time.perf_counter() - t0) * 1e3

    def _warmup(self) -> None:
        """Run each model once on a black frame so the first real frame isn't 100+ ms late."""
        self._rgb.fill(0)
        image = self._tasks.wrap(self._rgb)
        for key in TASK_KEYS:
            self._call(key, image, self._ts(key, 0.0), 0.0)
        self._failed.clear()
        self.errors = {k: 0 for k in TASK_KEYS}
