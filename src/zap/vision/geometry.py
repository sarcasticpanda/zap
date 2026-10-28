"""Geometry shared by gesture detection and effects.

Distances are expressed relative to body size so thresholds work at any
distance from the camera: hand features in units of hand size ``s`` (wrist to
middle-finger knuckle) and face-relative positions in units of face width
``fw``. In the face frame, u grows toward image-right and v grows DOWN the face
(negative v is above the nose bridge).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from zap.signals import (
    FACE_BROW_PEAKS,
    FACE_CHIN,
    FACE_FOREHEAD_TOP,
    FACE_LEFT_EDGE,
    FACE_LOWER_LIDS,
    FACE_NOSE_BRIDGE,
    FACE_RIGHT_EDGE,
    FINGER_PIPS,
    FINGERTIPS,
    MIDDLE_TIP,
    WRIST,
    Face,
    Hand,
)


# ---------------------------------------------------------------- filters
class OneEuro:
    """One Euro filter (Casiez et al.) for scalars or numpy vectors."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 1.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x = None
        self._dx = None
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x = self._dx = self._t = None

    def __call__(self, x, t: float):
        x = np.asarray(x, dtype=np.float64)
        if self._t is None or t <= self._t:
            self._x, self._dx, self._t = x, np.zeros_like(x), t
            return x.copy()
        dt = t - self._t
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * dt))
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x.copy()


class BlendEMA:
    """Exponential smoothing of blendshape scores (alpha 0.4 by default)."""

    def __init__(self, alpha: float = 0.4) -> None:
        self.alpha = alpha
        self.values: dict[str, float] = {}

    def update(self, scores: dict[str, float]) -> dict[str, float]:
        a = self.alpha
        for k, v in scores.items():
            self.values[k] = v if k not in self.values else a * v + (1 - a) * self.values[k]
        return self.values

    def get(self, key: str, default: float = 0.0) -> float:
        return self.values.get(key, default)


# ---------------------------------------------------------------- hands
def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))


def fingers_extended(hand: Hand, ratio: float = 1.10) -> np.ndarray:
    """Bool[4] for index, middle, ring, pinky: |tip - wrist| > ratio * |pip - wrist|."""
    w = hand.lm[WRIST, :2]
    tips = np.linalg.norm(hand.lm[list(FINGERTIPS), :2] - w, axis=1)
    pips = np.linalg.norm(hand.lm[list(FINGER_PIPS), :2] - w, axis=1)
    return tips > ratio * pips


def is_open(hand: Hand, min_fingers: int = 3) -> bool:
    return int(fingers_extended(hand).sum()) >= min_fingers or (
        hand.gesture == "Open_Palm" and hand.gesture_score >= 0.5
    )


def is_fist(hand: Hand) -> bool:
    if hand.gesture == "Closed_Fist" and hand.gesture_score >= 0.6:
        return True
    w = hand.lm[WRIST, :2]
    tips = np.linalg.norm(hand.lm[list(FINGERTIPS), :2] - w, axis=1)
    pips = np.linalg.norm(hand.lm[list(FINGER_PIPS), :2] - w, axis=1)
    return bool(np.all(tips < 0.95 * pips))


def is_upright(hand: Hand, k: float = 0.9) -> bool:
    """Middle fingertip at least k*s above the wrist (image y grows down)."""
    return float(hand.lm[WRIST, 1] - hand.lm[MIDDLE_TIP, 1]) >= k * hand.size


def hand_bbox(hand: Hand) -> tuple[float, float, float, float]:
    x0, y0 = hand.lm[:, :2].min(axis=0)
    x1, y1 = hand.lm[:, :2].max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Area of a∩b divided by the area of a."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


# ---------------------------------------------------------------- face frame
@dataclass
class FaceFrame:
    origin: np.ndarray  # nose bridge, pixels
    ex: np.ndarray  # unit vector along the eye line, toward image-right
    ey: np.ndarray  # unit vector down the face
    fw: float  # face width in pixels

    def to_uv(self, pts: np.ndarray) -> np.ndarray:
        """(N, 2+) pixel points -> (N, 2) face-frame coordinates in face widths."""
        d = np.asarray(pts, dtype=np.float64)[..., :2] - self.origin
        return np.stack([d @ self.ex, d @ self.ey], axis=-1) / self.fw

    def v_of(self, face: Face, idx: int) -> float:
        return float(self.to_uv(face.lm[idx : idx + 1])[0, 1])


def face_frame(face: Face) -> FaceFrame:
    left = face.lm[FACE_LEFT_EDGE, :2].astype(np.float64)
    right = face.lm[FACE_RIGHT_EDGE, :2].astype(np.float64)
    ex = right - left
    fw = float(np.linalg.norm(ex)) or 1.0
    ex = ex / fw
    ey = np.array([-ex[1], ex[0]])  # rotate +90deg: points down the face in image coords
    return FaceFrame(face.lm[FACE_NOSE_BRIDGE, :2].astype(np.float64), ex, ey, fw)


def brow_v(face: Face, ff: FaceFrame) -> float:
    return float(np.mean([ff.v_of(face, i) for i in FACE_BROW_PEAKS]))


def eye_v(face: Face, ff: FaceFrame) -> float:
    return float(np.mean([ff.v_of(face, i) for i in FACE_LOWER_LIDS]))


def forehead_top_v(face: Face, ff: FaceFrame) -> float:
    return ff.v_of(face, FACE_FOREHEAD_TOP)


def chin_v(face: Face, ff: FaceFrame) -> float:
    return ff.v_of(face, FACE_CHIN)


def head_yaw_deg(face: Face) -> float:
    """Approximate yaw: from the transformation matrix if present, else nose offset between face edges."""
    if face.matrix is not None and face.matrix.shape == (4, 4):
        r = face.matrix[:3, :3]
        return float(math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2]))))
    left, right = face.lm[FACE_LEFT_EDGE, 0], face.lm[FACE_RIGHT_EDGE, 0]
    nose = face.lm[FACE_NOSE_BRIDGE, 0]
    span = max(1e-6, right - left)
    return float(((nose - left) / span - 0.5) * 120.0)


# ---------------------------------------------------------------- motion
class HandTracks:
    """Associates hands across frames (nearest palm centre) to estimate palm speed in px/s."""

    def __init__(self, max_jump: float = 250.0) -> None:
        self.max_jump = max_jump
        self._prev: list[tuple[np.ndarray, float]] = []
        self.speeds: list[float] = []

    def update(self, hands: list[Hand], t: float) -> list[float]:
        speeds = []
        current = []
        for h in hands:
            pc = h.palm_center
            best, best_d = None, self.max_jump
            for p, pt in self._prev:
                d = float(np.linalg.norm(pc - p))
                if d < best_d:
                    best, best_d = (p, pt), d
            if best is None or t <= best[1]:
                speeds.append(0.0)
            else:
                speeds.append(best_d / (t - best[1]))
            current.append((pc, t))
        self._prev = current
        self.speeds = speeds
        return speeds
