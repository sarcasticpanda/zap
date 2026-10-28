"""Data passed from perception to gestures and effects.

All landmark coordinates are in OUTPUT pixel space (default 1280x720):
x and y are pixels, z is scaled by the frame width like x. Frames are never
mirrored, so "left" and "right" are as the meeting sees them. MediaPipe's
handedness labels are deliberately not used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Hand landmark indices (MediaPipe hand model).
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20
FINGERTIPS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
FINGER_PIPS = (INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP)
PALM_POINTS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

# Face landmark indices (MediaPipe 478-point face mesh).
FACE_FOREHEAD_TOP = 10
FACE_MID_FOREHEAD = 151
FACE_GLABELLA = 9
FACE_NOSE_BRIDGE = 168
FACE_NOSE_TIP = 1
FACE_CHIN = 152
FACE_LEFT_EDGE = 234
FACE_RIGHT_EDGE = 454
FACE_BROW_PEAKS = (105, 334)
FACE_EYE_CORNERS = (33, 133, 362, 263)
FACE_LOWER_LIDS = (145, 374)
FACE_MOUTH_CORNERS = (61, 291)
FACE_IRIS_CENTERS = (468, 473)


@dataclass
class Hand:
    lm: np.ndarray  # (21, 3) float32, output pixels
    score: float  # detection/handedness confidence
    gesture: str = "None"  # MediaPipe canned gesture name
    gesture_score: float = 0.0

    @property
    def size(self) -> float:
        """Hand scale s = |wrist - middle MCP| in pixels."""
        return float(np.linalg.norm(self.lm[MIDDLE_MCP, :2] - self.lm[WRIST, :2]))

    @property
    def palm_center(self) -> np.ndarray:
        return self.lm[list(PALM_POINTS), :2].mean(axis=0)


@dataclass
class Face:
    lm: np.ndarray  # (478, 3) float32, output pixels
    blend: dict[str, float] = field(default_factory=dict)  # blendshape name -> score
    matrix: np.ndarray | None = None  # 4x4 facial transformation matrix
    t: float = 0.0  # perf_counter time of the frame this face came from
    fresh: bool = True  # False when reused from the short-lived face cache

    @property
    def width(self) -> float:
        """Face width fw = |left edge - right edge| in pixels."""
        return float(np.linalg.norm(self.lm[FACE_RIGHT_EDGE, :2] - self.lm[FACE_LEFT_EDGE, :2]))

    @property
    def center(self) -> np.ndarray:
        return self.lm[FACE_NOSE_BRIDGE, :2].copy()

    def bbox(self) -> tuple[int, int, int, int]:
        x0, y0 = self.lm[:, :2].min(axis=0)
        x1, y1 = self.lm[:, :2].max(axis=0)
        return int(x0), int(y0), int(x1), int(y1)


@dataclass
class Signals:
    """Everything perception learned about one camera frame."""

    t: float  # perf_counter time the frame was captured
    seq: int
    hands: list[Hand] = field(default_factory=list)
    face: Face | None = None
    mask: np.ndarray | None = None  # float32 person confidence at inference resolution, or None
    mask_fresh: bool = False  # True when the mask was computed for this very frame
    width: int = 1280
    height: int = 720


@dataclass
class FramePacket:
    frame: np.ndarray  # BGR uint8 at output resolution
    t: float
    seq: int
