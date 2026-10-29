"""Raw per-frame truths for the held (non-snap) gestures.

Each check answers only "does this frame look like the pose?". Holding,
hysteresis and priorities belong to ``dwell.py`` and ``arbiter.py``. Positions
are measured in the face frame (face widths, v down the face) and speeds in
face widths per second, so one set of thresholds works near and far from the
camera.

Several things look like poses but aren't, and are rejected here: hands moving
fast (talking with your hands), an overhead stretch, a facepalm over the eyes,
scratching your forehead, and brows that simply sit high on someone's face.

Two rules come from recordings of a real user. A hand laid flat on the forehead
is often LOST by the hand model (it merges with the face) while the face stays
tracked, so a hand that was resting on the forehead and vanished there still
counts as on the forehead for a while. And brows sit noticeably higher while
someone concentrates on another gesture, so the eyebrow raise is measured
against that person's own resting level and is blocked whenever a hand is in
view.
"""

from __future__ import annotations

import difflib
import logging
import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any, NamedTuple

import numpy as np

from zap.config import ConfigError
from zap.gestures.dwell import DwellParams
from zap.gestures.types import (
    CANCEL,
    COLLAR,
    COVER_FACE,
    EYEBROWS_UP,
    FOREHEAD,
    HAND_SIGNS,
    HANDS_UP,
    PEACE,
    POINT_UP,
    ROCK,
    THUMBS_DOWN,
    THUMBS_UP,
    TWO_THUMBS,
)
from zap.signals import (
    FACE_BROW_PEAKS,
    FACE_CHIN,
    FACE_FOREHEAD_TOP,
    FACE_LOWER_LIDS,
    FINGER_PIPS,
    FINGERTIPS,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_TIP,
    THUMB_TIP,
    WRIST,
    Face,
    Hand,
    Signals,
)
from zap.vision.geometry import (
    BlendEMA,
    FaceFrame,
    HandTracks,
    dist,
    face_frame,
    fingers_extended,
    hand_bbox,
    head_yaw_deg,
    is_open,
    is_upright,
    overlap_ratio,
)

log = logging.getLogger(__name__)

POSE_NAMES = (HANDS_UP, TWO_THUMBS, PEACE, *HAND_SIGNS, FOREHEAD, COVER_FACE, EYEBROWS_UP, COLLAR, CANCEL, THUMBS_UP)
STATIC_POSES = (HANDS_UP, TWO_THUMBS, PEACE, *HAND_SIGNS, FOREHEAD, EYEBROWS_UP, COLLAR,  # blanked while hands
                THUMBS_UP)  # move fast
GENERAL = "general"
_TIPS = list(FINGERTIPS)


# ---------------------------------------------------------------- parameters
@dataclass
class GeneralParams:
    min_score: float = 0.6  # hand detection confidence
    min_size: float = 0.045  # hand size s, fraction of frame width
    max_wrist_y: float = 1.2  # wrist below this fraction of frame height = hand too far out of the frame
    # (sitting close, the wrist is often just below the picture during collar / cover-face: still usable)
    edge_margin: float = 0.005  # landmarks must stay this far inside the left/right edges (fraction of W)
    talk_speed: float = 3.0  # fw/s; any palm faster than this blanks the static poses
    face_hold_s: float = 0.5  # reuse the last face this long for hand+face poses
    snap_pinch: float = 0.31  # |thumb tip - middle tip| / s below this = loaded snap
    snap_reach: float = 1.11  # ... with |middle tip - wrist| / s above this


@dataclass
class _Dwelled:
    t_on: float = 0.5
    t_off: float = 0.5
    min_show: float = 0.0
    cooldown: float = 8.0
    rearm: float = 0.4
    ratio: float = 0.8
    dropout: float = 0.12

    def dwell(self) -> DwellParams:
        return DwellParams(self.t_on, self.t_off, self.min_show, self.cooldown, self.rearm, self.ratio, self.dropout)


@dataclass
class HandsUpParams(_Dwelled):
    t_on: float = 0.4
    t_off: float = 0.6
    min_show: float = 2.5
    cooldown: float = 4.0
    min_abs_u: float = 0.6  # palms this far left/right of the nose bridge
    above_top: float = 1.0  # palms no higher than forehead top - this (rejects an overhead stretch)
    below_chin: float = 0.4  # palms no lower than chin + this
    min_gap: float = 1.6  # palm-to-palm distance
    max_speed: float = 2.5
    upright_k: float = 0.9


@dataclass
class ForeheadParams(_Dwelled):
    t_on: float = 0.7
    t_off: float = 0.5
    cooldown: float = 8.0
    t_on_fast: float = 0.5  # used when the face already looks pained
    fast_brow_inner: float = 0.30
    fast_frown: float = 0.20
    half_width: float = 0.55  # forehead region: |u| <= this
    above_top: float = 0.30  # v >= forehead top - this
    below_brow: float = 0.02  # v <= brow - this
    min_tips: int = 2
    palm_expand: float = 0.15  # palm centre may sit in the region grown by this instead
    min_size: float = 0.45  # s / fw
    max_size: float = 1.6
    max_speed: float = 0.8  # fw/s, palm speed over speed_window_s (single frames jitter too much)
    speed_window_s: float = 0.25
    max_tip_spread: float = 1.5  # fw/s of fingertip motion relative to the palm (scratching)
    spread_window_s: float = 0.5
    # A hand resting on the forehead that the hand model then loses still counts (see module doc).
    occl_half_width: float = 0.7  # region the hand vanished in: |u| <= this ...
    occl_above_top: float = 0.5  # ... v from forehead top - this down to the eyes
    occl_hold_s: float = 6.0  # assume the hand stays at most this long
    occl_edge: float = 0.05  # a hand last seen this close to the frame border left the frame
    occl_max_speed: float = 1.0  # fw/s; a hand moving through the region was not resting on it
    occl_min_seen_s: float = 0.1  # in the region at least this long before vanishing
    occl_pair_s: float = 0.3  # two hands vanishing over the face this close together = cover_face
    occl_face_s: float = 0.5  # the face must have been fresh this recently


@dataclass
class CoverFaceParams(_Dwelled):
    t_on: float = 0.0  # the detector already emits a one-frame truth at the drop
    t_off: float = 0.0
    cooldown: float = 10.0
    min_overlap: float = 0.3  # each hand bbox overlaps the face bbox by this share of its own area
    palm_margin: float = 0.25  # ... and its palm centre sits inside the face bbox grown by this share of its width
    hold_s: float = 0.5  # covered at least this long ...
    drop_window_s: float = 2.0  # ... then the face must be back within this of uncovering
    dropout_s: float = 0.35  # short losses of the covering hands are tolerated
    bbox_memory_s: float = 3.0  # use the last face bbox this long after the face disappears


@dataclass
class EyebrowsParams(_Dwelled):
    """Brow level = mean of browInnerUp, browOuterUpLeft, browOuterUpRight (raw, per fresh face).

    Measured on a real user: ~0.20 relaxed, 0.85-0.96 raised on purpose, but 0.45-0.65
    while concentrating on other gestures - hence the per-user baseline and the hand block.
    """

    t_on: float = 0.6
    t_off: float = 0.5
    cooldown: float = 6.0
    min_level: float = 0.7  # the level must reach this (concentrating on other gestures: 0.54-0.68) ...
    above_base: float = 0.4  # ... and the resting baseline plus this ...
    max_level: float = 0.82  # ... capped here: the user's deliberate raises measured 0.83-0.97
    start_base: float = 0.2  # baseline until enough samples are in
    base_pct: float = 20.0  # baseline = this percentile of the level ...
    baseline_window_s: float = 60.0  # ... over this long, on frames with a fresh face and no hand
    baseline_every_s: float = 0.1
    baseline_min_samples: int = 20
    max_asym: float = 0.35  # |outer left - outer right|: one brow up is not the gesture (real raises <= 0.22)
    max_yaw: float = 15.0
    max_jaw: float = 0.3
    hand_clear_s: float = 0.4  # a hand near the face blocks it, and for this long after
    hand_near_fw: float = 1.6  # "near": palm centre within this many face widths of the face centre
    speak_var: float = 0.004  # jawOpen variance over speak_window_s above this = speaking
    speak_window_s: float = 1.0
    speak_factor: float = 1.5  # hold this many times longer while speaking
    stale_s: float = 0.5  # forget the brow verdict when no fresh face for this long
    max_tilt: float = 0.06  # head tilted up/down from your usual pose by more than this (nose height ratio)
    recent_s: float = 10.0  # calibrated: the resting level is the median of the last this-many seconds ...
    rise_share: float = 0.6  # ... and a raise must lift the brows by this share of your calibrated rise
    # also lifts the brow readings: not a raise


@dataclass
class CollarParams(_Dwelled):
    t_on: float = 0.6
    t_off: float = 0.5
    cooldown: float = 8.0
    pinch: float = 0.35  # |thumb tip - index tip| / s
    half_width: float = 0.9  # collar region: |u| <= this
    top: float = 0.05  # v in [chin + top, chin + bottom]
    bottom: float = 0.7
    min_tips: int = 2
    max_speed: float = 1.2


@dataclass
class CancelParams(_Dwelled):
    t_on: float = 0.8
    t_off: float = 0.2
    cooldown: float = 0.0
    min_fingers: int = 4
    open_palm: float = 0.7


@dataclass
class ThumbsUpParams(_Dwelled):
    t_on: float = 0.6
    t_off: float = 0.3
    cooldown: float = 2.0
    min_score: float = 0.75


@dataclass
class TwoThumbsParams(_Dwelled):
    """A thumbs-up (MediaPipe's "Thumb_Up", or the shape), held. One hand is enough by default: two
    at once proved fiddly for the hand model, and a thumbs-up in a meeting is a fine "deal!" moment."""

    t_on: float = 0.7
    t_off: float = 0.4
    cooldown: float = 5.0
    min_score: float = 0.45  # MediaPipe "Thumb_Up" confidence
    min_hand_score: float = 0.5
    min_hands: int = 1


def brow_height(face: Face) -> float:
    """How far the brows sit above the upper eyelids, in eye-corner distances (rises ~15-25% when
    raised). Read from the face mesh, so it still works when glasses confuse the brow scores."""
    lm = face.lm
    d = 0.5 * ((lm[159, 1] - lm[105, 1]) + (lm[386, 1] - lm[334, 1]))
    return float(d / max(1.0, float(np.hypot(*(lm[263, :2] - lm[33, :2])))))


def _nose_ratio(face: Face) -> float:
    """Where the nose tip sits between the eyes and the chin (0 = eye level, 1 = chin): it moves when
    the head tilts up or down, which also makes the brows look raised to the face model."""
    lm = face.lm
    eyes = 0.5 * (lm[33, 1] + lm[263, 1])
    return float((lm[1, 1] - eyes) / max(1.0, lm[152, 1] - eyes))


@dataclass
class PeaceParams(_Dwelled):
    """A peace sign (MediaPipe's canned "Victory"), held: "To Be Continued"."""

    t_on: float = 0.8
    t_off: float = 0.4
    cooldown: float = 10.0
    min_score: float = 0.5


@dataclass
class HandSignParams(_Dwelled):
    """A one-hand sign (MediaPipe's canned gesture, ``SIGN_LABELS``), held still, away from the face.

    The classifier's label must agree with the finger shape (``SIGN_SHAPES``),
    unless it is very sure. Hands at the face are ignored: an index finger along
    the cheek (chin on hand) or at the lips ("shh") is not a sign.
    """

    t_on: float = 0.6
    t_off: float = 0.4
    cooldown: float = 6.0
    dropout: float = 0.2  # the classifier flickers to "None" for a frame or two while a sign is held
    min_score: float = 0.5  # label confidence when the finger shape agrees ...
    sure_score: float = 0.8  # ... or this much on its own
    max_speed: float = 1.5  # fw/s: a hand still moving is on its way somewhere
    max_face_overlap: float = 0.5  # share of the hand's box that may cover the face's box (a sign held
    # right beside the cheek overlaps it a little; a fingertip ON the face is rejected regardless)
    shape_only: bool = False  # the finger shape alone counts while the classifier sees no other sign


@dataclass
class PointUpParams(HandSignParams):
    t_on: float = 0.7  # a finger raised to make a point in conversation is quicker than this


@dataclass
class RockParams(HandSignParams):
    """MediaPipe only knows the thumb-out version (🤟 "ILoveYou"); 🤘 comes back as "None", so the shape
    has to do. On real photos of 26 hand gestures the shape matched 🤘 and nothing else."""

    shape_only: bool = True


@dataclass
class ArbiterParams:
    """Timing rules between gestures (``[gestures.arbiter]``)."""

    settle_s: float = 0.1  # a pose that just turned "on" waits this long for a higher-priority one
    gap_s: float = 1.0  # between two different pose effects, and after the app stops being busy
    snap_hold_s: float = 0.6  # a loaded snap pinch silences hand poses this long
    pending_max_s: float = 1.5  # a pose that can't fire by then is dropped (and must be released)
    charge_timeout_s: float = 1.5  # forget a snap "charge" with no burst or discharge after this


ARBITER = "arbiter"

# MediaPipe canned gesture per hand sign
SIGN_LABELS = {POINT_UP: "Pointing_Up", THUMBS_DOWN: "Thumb_Down", ROCK: "ILoveYou"}

PARAM_TYPES: dict[str, type] = {
    GENERAL: GeneralParams,
    ARBITER: ArbiterParams,
    HANDS_UP: HandsUpParams,
    FOREHEAD: ForeheadParams,
    COVER_FACE: CoverFaceParams,
    EYEBROWS_UP: EyebrowsParams,
    COLLAR: CollarParams,
    CANCEL: CancelParams,
    THUMBS_UP: ThumbsUpParams,
    TWO_THUMBS: TwoThumbsParams,
    PEACE: PeaceParams,
    POINT_UP: PointUpParams,
    THUMBS_DOWN: HandSignParams,
    ROCK: RockParams,
}


def _did_you_mean(key: str, options) -> str:
    match = difflib.get_close_matches(key, list(options), n=1)
    return f" (did you mean '{match[0]}'?)" if match else ""


def _apply(params: Any, values: Mapping[str, Any], where: str) -> Any:
    names = [f.name for f in fields(params)]
    changes = {}
    for key, value in values.items():
        if key not in names:
            raise ConfigError(f"unknown setting 'gestures.{where}.{key}'{_did_you_mean(key, names)}")
        expected = type(getattr(params, key))
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not isinstance(value, expected) or (expected is not bool and isinstance(value, bool)):
            raise ConfigError(f"'gestures.{where}.{key}' should be {expected.__name__}, got {value!r}")
        changes[key] = value
    return replace(params, **changes)


def load_params(overrides: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Default parameter dataclasses per gesture, with ``config.gestures`` overrides applied.

    Unknown gestures or keys raise ConfigError (a ValueError) with a did-you-mean hint.
    """
    out = {name: cls() for name, cls in PARAM_TYPES.items()}
    for name, values in (overrides or {}).items():
        if name not in PARAM_TYPES:
            raise ConfigError(f"unknown gesture '[gestures.{name}]'{_did_you_mean(name, PARAM_TYPES)}")
        if not isinstance(values, Mapping):
            raise ConfigError(f"'gestures.{name}' must be a [section]")
        out[name] = _apply(out[name], values, name)
    return out


# ---------------------------------------------------------------- small helpers
def usable(hand: Hand, width: int, height: int, g: GeneralParams) -> bool:
    """Confident, big enough, and not cut off by the bottom or side of the frame."""
    if hand.score < g.min_score or hand.size < g.min_size * width:
        return False
    if hand.lm[WRIST, 1] > g.max_wrist_y * height:
        return False
    xs = hand.lm[:, 0]
    return bool(xs.min() >= g.edge_margin * width and xs.max() <= (1.0 - g.edge_margin) * width)


def thumb_up_shape(h: Hand) -> bool:
    """Fist with the thumb pointing up: the thumb tip is the highest point, well above the
    curled fingers, and at least three fingertips are folded back toward the wrist."""
    p = h.lm[:, :2]
    s = float(np.hypot(*(p[WRIST] - p[MIDDLE_MCP])))
    if s < 1e-6:
        return False
    tips_y = p[list(FINGERTIPS), 1]
    up = p[THUMB_TIP, 1] < p[3, 1] < p[2, 1] and p[2, 1] - p[THUMB_TIP, 1] > 0.45 * s
    above = p[THUMB_TIP, 1] < tips_y.min() - 0.25 * s
    return bool(up and above and _curled(p) >= 3)


def thumb_down_shape(h: Hand) -> bool:
    """Fist with the thumb pointing down: :func:`thumb_up_shape` upside down."""
    p = h.lm[:, :2]
    s = float(np.hypot(*(p[WRIST] - p[MIDDLE_MCP])))
    if s < 1e-6:
        return False
    tips_y = p[list(FINGERTIPS), 1]
    down = p[THUMB_TIP, 1] > p[3, 1] > p[2, 1] and p[THUMB_TIP, 1] - p[2, 1] > 0.45 * s
    below = p[THUMB_TIP, 1] > tips_y.max() + 0.25 * s
    return bool(down and below and _curled(p) >= 3)


def point_up_shape(h: Hand) -> bool:
    """Index finger up (within ~35 degrees of vertical), the other three folded."""
    p = h.lm[:, :2]
    s = float(np.hypot(*(p[WRIST] - p[MIDDLE_MCP])))
    ext = fingers_extended(h)
    if s < 1e-6 or not ext[0] or ext[1:].any():
        return False
    d = p[INDEX_TIP] - p[INDEX_MCP]
    return bool(-d[1] > 0.6 * s and abs(d[0]) < 0.7 * -d[1])


def rock_shape(h: Hand) -> bool:
    """Index and pinky up, middle and ring folded; the thumb may stick out (🤟) or not (🤘)."""
    p = h.lm[:, :2]
    s = float(np.hypot(*(p[WRIST] - p[MIDDLE_MCP])))
    ext = fingers_extended(h)
    if s < 1e-6 or not (ext[0] and ext[3]) or ext[1] or ext[2]:
        return False
    return bool(p[WRIST, 1] - max(p[INDEX_TIP, 1], p[PINKY_TIP, 1]) > 0.6 * s)


def _curled(p: np.ndarray) -> int:
    """How many of the four fingers are folded (tip nearer the wrist than its middle joint)."""
    return int(sum(np.hypot(*(p[t] - p[WRIST])) < 1.05 * np.hypot(*(p[q] - p[WRIST]))
                   for t, q in zip(FINGERTIPS, FINGER_PIPS, strict=True)))


SIGN_SHAPES = {POINT_UP: point_up_shape, THUMBS_DOWN: thumb_down_shape, ROCK: rock_shape}
SIGN_TIPS = {POINT_UP: (INDEX_TIP,), THUMBS_DOWN: (THUMB_TIP,), ROCK: (INDEX_TIP, PINKY_TIP)}  # off the face


def snap_loaded_hands(sig: Signals, pinch: float = 0.31, reach: float = 1.11, min_score: float = 0.6) -> list[np.ndarray]:
    """Palm centres of hands in a loaded snap (thumb pressed to an extended middle finger)."""
    out = []
    for h in sig.hands:
        s = h.size
        if h.score < min_score or s <= 1e-6:
            continue
        if dist(h.lm[THUMB_TIP], h.lm[MIDDLE_TIP]) / s < pinch and dist(h.lm[MIDDLE_TIP], h.lm[WRIST]) / s > reach:
            out.append(h.palm_center)
    return out


class _Levels(NamedTuple):
    """Face-frame v of the forehead top, brows, lower lids and chin (geometry's *_v helpers, in one transform)."""

    top: float
    brow: float
    eye: float
    chin: float


_LEVEL_IDX = [FACE_FOREHEAD_TOP, *FACE_BROW_PEAKS, *FACE_LOWER_LIDS, FACE_CHIN]


def _levels(face: Face, ff: FaceFrame) -> _Levels:
    v = ff.to_uv(face.lm[_LEVEL_IDX])[:, 1]
    return _Levels(float(v[0]), float(v[1:3].mean()), float(v[3:5].mean()), float(v[5]))


def _in_region(uv: np.ndarray, half_width: float, v_lo: float, v_hi: float) -> np.ndarray:
    return (np.abs(uv[..., 0]) <= half_width) & (uv[..., 1] >= v_lo) & (uv[..., 1] <= v_hi)


def _box_gap(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Distance between two (x0, y0, x1, y1) boxes; 0 when they touch or overlap."""
    dx = max(0.0, a[0] - b[2], b[0] - a[2])
    dy = max(0.0, a[1] - b[3], b[1] - a[3])
    return math.hypot(dx, dy)


def _inside(pt: np.ndarray, box: tuple[float, ...]) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


@dataclass
class _Track:
    """One hand followed across frames (nearest palm centre)."""

    pc: np.ndarray  # palm centre, px
    tips: np.ndarray  # fingertips, px
    t: float  # last frame it was seen
    edge: bool = False  # some landmark was near the frame border
    path: deque = field(default_factory=deque)  # (t, palm centre)
    spread: deque = field(default_factory=deque)  # (t, mean squared fingertip speed relative to the palm)
    region_since: float | None = None  # entered the forehead occlusion region at this time
    speed_fw: float = 0.0  # windowed palm speed, fw/s (set by the detector)


class _Tracks:
    """Hands followed across frames: palm speed over a window, fingertip motion relative to the
    palm (scratching), and which hands vanished or newly appeared this frame."""

    def __init__(self, max_jump: float = 250.0, keep_s: float = 1.0) -> None:
        self.max_jump, self.keep_s = max_jump, keep_s
        self.current: list[_Track] = []
        self.vanished: list[_Track] = []
        self.new: list[int] = []

    def update(self, hands: list[Hand], t: float, width: int, height: int, edge: float) -> None:
        prev, claimed = self.current, set()
        current, new = [], []
        x_lo, x_hi, y_lo, y_hi = edge * width, (1 - edge) * width, edge * height, (1 - edge) * height
        for i, h in enumerate(hands):
            pc, tips = h.palm_center.astype(np.float64), h.lm[_TIPS, :2].astype(np.float64)
            best, best_d = -1, self.max_jump
            for j, tr in enumerate(prev):
                d = float(np.linalg.norm(pc - tr.pc))
                if j not in claimed and d < best_d:
                    best, best_d = j, d
            xs, ys = h.lm[:, 0], h.lm[:, 1]
            near_edge = bool(xs.min() < x_lo or xs.max() > x_hi or ys.min() < y_lo or ys.max() > y_hi)
            if best >= 0:
                claimed.add(best)
                tr = prev[best]
                if t > tr.t:
                    rel = ((tips - tr.tips) - (pc - tr.pc)) / (t - tr.t)
                    tr.spread.append((t, float(np.mean(np.sum(rel * rel, axis=1)))))
                tr.pc, tr.tips, tr.t, tr.edge = pc, tips, t, near_edge
            else:
                tr = _Track(pc, tips, t, near_edge)
                new.append(i)
            tr.path.append((t, pc))
            for q in (tr.path, tr.spread):
                while q and t - q[0][0] > self.keep_s:
                    q.popleft()
            current.append(tr)
        self.vanished = [tr for j, tr in enumerate(prev) if j not in claimed]
        self.current, self.new = current, new

    def spread(self, i: int, now: float, window: float) -> float:
        """RMS fingertip speed relative to the palm over the last ``window`` s, px/s (0 if unknown)."""
        if i >= len(self.current):
            return 0.0
        ms = [m for t, m in self.current[i].spread if now - t <= window]
        return math.sqrt(sum(ms) / len(ms)) if len(ms) >= 3 else 0.0

    def speed(self, i: int, now: float, window: float) -> float:
        """Palm speed over the last ``window`` s, px/s. Steadier than frame-to-frame differences,
        which landmark jitter alone pushes past 1 fw/s now and then."""
        if i >= len(self.current):
            return 0.0
        path = self.current[i].path
        if len(path) < 2:
            return 0.0
        t1, p1 = path[-1]
        t0, p0 = next(((t, pc) for t, pc in path if now - t <= window + 1e-6), path[-2])
        if t1 - t0 <= 1e-6:  # window shorter than one frame
            t0, p0 = path[-2]
        return float(np.linalg.norm(p1 - p0)) / max(t1 - t0, 1e-6)


class _CoverWatch:
    """Both hands over the face for a while, then dropped: fires once as the face comes back."""

    def __init__(self, p: CoverFaceParams) -> None:
        self.p = p
        self.covering = False
        self.held = 0.0
        self._since = 0.0
        self._last = 0.0

    def update(self, covered: bool, face_back: bool, now: float) -> bool:
        p = self.p
        if covered:
            if not self.covering:
                self._since = now
            self.covering, self._last = True, now
            self.held = now - self._since
            return False
        if not self.covering:
            return False
        gap = now - self._last
        if self._last - self._since + 1e-6 >= p.hold_s:
            if gap > p.drop_window_s:
                self.covering = False
            elif face_back:
                self.covering = False
                return True
        elif gap > p.dropout_s:
            self.covering = False
        return False


class _BrowWatch:
    """Brow level against a per-user resting baseline, plus a speaking detector.

    level = mean(browInnerUp, browOuterUpLeft, browOuterUpRight) of the last fresh face;
    baseline = ``base_pct`` percentile of the level over the last minute of hand-free frames;
    raised when level >= min(max_level, max(min_level, baseline + above_base)).
    """

    def __init__(self, p: EyebrowsParams) -> None:
        self.p = p
        self.ema = BlendEMA(0.4)  # smoothed scores, only for the "pained face" forehead shortcut
        self.raised = False
        self.speaking = False
        self.seen = -math.inf
        self.level = 0.0
        self.asym = 0.0
        self.base = p.start_base
        self.rise: float | None = None  # how much your brows rise, from a calibration (Ctrl+Alt+C)
        self._recent: deque[tuple[float, float]] = deque()  # (t, value) of the calibrated signal, hands clear
        self.signal = "blend"  # "blend" (face-model brow scores) or "geo" (brow height over the eyelids)
        self.geo = 0.0
        self.threshold = self._threshold()
        self._base: deque[tuple[float, float]] = deque()
        self._last_sample = -math.inf
        self._jaw: deque[tuple[float, float]] = deque()
        self.tilt = 0.0
        self._tilts: deque[float] = deque(maxlen=600)

    def update(self, face: Face, now: float, hands_clear: bool) -> None:
        """Call with fresh faces only; cached faces carry stale blendshapes."""
        p = self.p
        b = face.blend
        self.ema.update(b)
        self.seen = now
        jaw = float(b.get("jawOpen", 0.0))
        self._jaw.append((now, jaw))
        while now - self._jaw[0][0] > p.speak_window_s:
            self._jaw.popleft()
        self.speaking = len(self._jaw) >= 5 and float(np.var([j for _, j in self._jaw])) > p.speak_var

        outer_l, outer_r = float(b.get("browOuterUpLeft", 0.0)), float(b.get("browOuterUpRight", 0.0))
        self.level = (float(b.get("browInnerUp", 0.0)) + outer_l + outer_r) / 3.0
        self.asym = abs(outer_l - outer_r)
        self.geo = brow_height(face)
        tilt = _nose_ratio(face)
        value = self.geo if self.signal == "geo" else self.level
        if hands_clear and now - self._last_sample >= p.baseline_every_s:
            self._sample(now)
            self._tilts.append(tilt)
            self._recent.append((now, value))
        while self._recent and now - self._recent[0][0] > p.recent_s:
            self._recent.popleft()
        if self.rise is not None and len(self._recent) >= 10:
            self.base = float(np.median([v for _, v in self._recent]))
        self.tilt = tilt - (float(np.median(self._tilts)) if len(self._tilts) >= 10 else tilt)
        self.threshold = self._threshold()
        frontal = abs(head_yaw_deg(face)) < p.max_yaw and jaw < p.max_jaw and abs(self.tilt) < p.max_tilt
        if self.rise is not None:
            self.raised = bool(frontal and hands_clear and value >= self.threshold
                               and (self.signal == "geo" or self.asym < p.max_asym))
        else:
            self.raised = bool(frontal and hands_clear and self.level >= self.threshold and self.asym < p.max_asym)

    def expire(self, now: float) -> None:
        if now - self.seen > self.p.stale_s:
            self.raised = self.speaking = False

    def _threshold(self) -> float:
        p = self.p
        if self.rise is not None:  # your resting level right now, plus most of your calibrated rise
            return self.base + p.rise_share * self.rise
        return min(p.max_level, max(p.min_level, self.base + p.above_base))

    def _sample(self, now: float) -> None:
        p = self.p
        self._last_sample = now
        self._base.append((now, self.level))
        while now - self._base[0][0] > p.baseline_window_s:
            self._base.popleft()
        if self.rise is None and len(self._base) >= p.baseline_min_samples:
            self.base = float(np.percentile(np.fromiter((v for _, v in self._base), np.float64), p.base_pct))


@dataclass
class _Occlusion:
    """A hand that vanished while resting on the forehead."""

    t0: float
    pc: np.ndarray


# ---------------------------------------------------------------- detector
class PoseDetector:
    """Per-frame raw pose truths. ``update`` once per frame, then feed the result to the Arbiter.

    After ``update``: ``t_on`` holds this frame's dwell times for gestures whose hold time
    adapts (forehead is quicker on a pained face; eyebrows need twice as long while
    speaking); ``snap_loaded`` lists palm centres of hands loaded for a snap; ``debug``
    holds measurements for the HUD.
    """

    def __init__(self, overrides: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self.params = load_params(overrides)
        self.reset()

    def reset(self) -> None:
        p = self.params
        self._tracks = HandTracks()
        self._hands = _Tracks()
        self._cover = _CoverWatch(p[COVER_FACE])
        self._brows = _BrowWatch(p[EYEBROWS_UP])
        self._face: Face | None = None
        self._face_seen = -math.inf
        self._face_fresh = -math.inf
        self._box: tuple[int, int, int, int] | None = None
        self._box_seen = -math.inf
        self._fw: float | None = None
        self._occl: _Occlusion | None = None
        self._face_vanish_t = -math.inf  # last time a hand vanished over the face
        self._hand_seen = -math.inf
        self.t_on: dict[str, float] = {FOREHEAD: p[FOREHEAD].t_on, EYEBROWS_UP: p[EYEBROWS_UP].t_on}
        self.snap_loaded: list[np.ndarray] = []
        self.debug: dict[str, Any] = {}

    @property
    def occluded(self) -> bool:
        """A hand is assumed to rest on the forehead although the hand model lost it."""
        return self._occl is not None

    def update(self, sig: Signals, now: float) -> dict[str, bool]:
        g: GeneralParams = self.params[GENERAL]
        hands = [h for h in sig.hands if usable(h, sig.width, sig.height, g)]
        face = self._face_ref(sig, now)
        if sig.face is not None and sig.face.fresh:
            self._face_fresh = now
        fw = self._ref_width(face, hands, sig.width)
        speeds = [s / fw for s in self._tracks.update(hands, now)]
        self._hands.update(hands, now, sig.width, sig.height, self.params[FOREHEAD].occl_edge)
        talking = any(s > g.talk_speed for s in speeds)

        on_face = self._hands_on_face(hands, sig, now)
        face_back = sig.face is not None and sig.face.fresh and on_face < 2  # the hands came off the face
        truths = dict.fromkeys(POSE_NAMES, False)
        truths[COVER_FACE] = self._cover.update(on_face >= 2, face_back, now)
        busy_face = self._cover.covering  # both hands on the face: no single-hand face poses
        self.debug = {"fw": round(fw, 1), "hands": len(hands), "speeds": [round(s, 2) for s in speeds],
                      "talking": talking, "covering": busy_face, "cover_held": round(self._cover.held, 2)}

        ff = lv = None
        if face is not None:
            ff = face_frame(face)
            lv = _levels(face, ff)
            self._mark_forehead_region(ff, lv, now)
        self._track_occlusion(sig, now, busy_face)
        if face is not None and not busy_face:
            truths[HANDS_UP] = self._hands_up(hands, lv, ff, speeds)
            truths[FOREHEAD] = self._forehead(hands, lv, ff, now) or self._occl is not None
            truths[COLLAR] = self._collar(hands, lv, ff, speeds)
        truths[EYEBROWS_UP] = self._eyebrows(sig, now) and not busy_face
        truths[CANCEL] = any(self._cancel(h) for h in sig.hands if h.score >= g.min_score)
        truths[THUMBS_UP] = any(self._thumbs_up(h) for h in hands)
        truths[TWO_THUMBS] = sum(self._thumb_up_any(h) for h in sig.hands) >= self.params[TWO_THUMBS].min_hands
        pp: PeaceParams = self.params[PEACE]
        truths[PEACE] = any(h.gesture == "Victory" and h.gesture_score >= pp.min_score
                            for h in sig.hands if h.score >= g.min_score)
        truths.update(self._hand_signs(hands, fw, now))
        if talking:
            for name in STATIC_POSES:
                truths[name] = False

        self.snap_loaded = self._loaded(sig, fw, g, ff, lv)
        self.debug["snap_loaded"] = len(self.snap_loaded)
        self.debug["occluded"] = self._occl is not None
        return truths

    # ------------------------------------------------------------ references
    def _face_ref(self, sig: Signals, now: float) -> Face | None:
        """This frame's face (fresh or cached by perception), else ours for ``face_hold_s``."""
        if sig.face is not None:
            self._face, self._face_seen = sig.face, now
            return sig.face
        if self._face is not None and now - self._face_seen <= self.params[GENERAL].face_hold_s:
            return self._face
        return None

    def _ref_width(self, face: Face | None, hands: list[Hand], width: int) -> float:
        """Face width in px for speed units; falls back to the last face, then hand size."""
        if face is not None and face.width > 1.0:
            self._fw = face.width
        if self._fw is not None:
            return self._fw
        if hands:
            return 1.5 * float(np.median([h.size for h in hands]))
        return 0.15 * width

    # ------------------------------------------------------------ poses
    def _hands_up(self, hands: list[Hand], lv: _Levels, ff: FaceFrame, speeds: list[float]) -> bool:
        p: HandsUpParams = self.params[HANDS_UP]
        if len(hands) != 2:
            return False
        if not all(is_open(h) and is_upright(h, p.upright_k) for h in hands):
            return False
        pcs = np.stack([h.palm_center for h in hands])
        uv = ff.to_uv(pcs)
        gap = float(np.linalg.norm(pcs[0] - pcs[1])) / ff.fw
        self.debug[HANDS_UP] = {"u": uv[:, 0].round(2).tolist(), "v": uv[:, 1].round(2).tolist(), "gap": round(gap, 2)}
        if uv[:, 0].min() > -p.min_abs_u or uv[:, 0].max() < p.min_abs_u:
            return False
        lo, hi = lv.top - p.above_top, lv.chin + p.below_chin
        if not bool(np.all((uv[:, 1] >= lo) & (uv[:, 1] <= hi))):
            return False
        return gap >= p.min_gap and max(speeds) < p.max_speed

    def _forehead(self, hands: list[Hand], lv: _Levels, ff: FaceFrame, now: float) -> bool:
        p: ForeheadParams = self.params[FOREHEAD]
        top, bottom, eyes = lv.top - p.above_top, lv.brow - p.below_brow, lv.eye
        for i, h in enumerate(hands):
            tips_in = int(_in_region(ff.to_uv(h.lm[_TIPS]), p.half_width, top, bottom).sum())
            pc = ff.to_uv(h.palm_center[None])[0]
            e = p.palm_expand
            palm_in = bool(_in_region(pc, p.half_width + e, top - e, bottom + e))
            spread = self._hands.spread(i, now, p.spread_window_s) / ff.fw
            speed = self._hands.current[i].speed_fw
            self.debug[FOREHEAD] = {"tips": tips_in, "palm_v": round(float(pc[1]), 2), "eye_v": round(eyes, 2),
                                    "size": round(h.size / ff.fw, 2), "speed": round(speed, 2),
                                    "spread": round(spread, 2)}
            if tips_in < p.min_tips and not palm_in:
                continue
            if pc[1] >= eyes:  # palm over the eyes: a facepalm, not a forehead pinch
                continue
            if not p.min_size <= h.size / ff.fw <= p.max_size or speed >= p.max_speed:
                continue
            if spread >= p.max_tip_spread:  # fingers busy relative to the palm: scratching
                continue
            return True
        return False

    def _mark_forehead_region(self, ff: FaceFrame, lv: _Levels, now: float) -> None:
        """Per tracked hand: windowed palm speed (fw/s) and since when it rests in the occlusion region."""
        p: ForeheadParams = self.params[FOREHEAD]
        for i, tr in enumerate(self._hands.current):
            tr.speed_fw = self._hands.speed(i, now, p.speed_window_s) / ff.fw
            u, v = ff.to_uv(tr.pc[None])[0]
            inside = abs(u) <= p.occl_half_width and lv.top - p.occl_above_top <= v <= lv.eye
            if not inside:
                tr.region_since = None
            elif tr.region_since is None:
                tr.region_since = now

    def _track_occlusion(self, sig: Signals, now: float, covering: bool) -> None:
        """Start, keep or end the "hand lost on the forehead" state (see the module docstring)."""
        p: ForeheadParams = self.params[FOREHEAD]
        tracks = self._hands
        face_ok = sig.face is not None and now - self._face_fresh <= p.occl_face_s
        over_face = [
            tr for tr in tracks.vanished
            if tr.region_since is not None or (self._box is not None and _inside(tr.pc, self._box))
        ]
        pair = len(over_face) >= 2 or (bool(over_face) and now - self._face_vanish_t <= p.occl_pair_s)
        if over_face:
            self._face_vanish_t = now
        if self._occl is not None:
            if tracks.new or covering or pair or not face_ok or now - self._occl.t0 > p.occl_hold_s:
                self._occl = None  # it came back (anywhere), both hands went (cover_face), or too long
            return
        if pair or covering or not face_ok or len(over_face) != 1:
            return
        tr = over_face[0]
        rested = tr.region_since is not None and tr.t - tr.region_since + 1e-6 >= p.occl_min_seen_s
        if rested and tr.speed_fw < p.occl_max_speed and not tr.edge:
            self._occl = _Occlusion(now, tr.pc)

    def _collar(self, hands: list[Hand], lv: _Levels, ff: FaceFrame, speeds: list[float]) -> bool:
        p: CollarParams = self.params[COLLAR]
        c = lv.chin
        v_lo, v_hi = c + p.top, c + p.bottom
        for i, h in enumerate(hands):
            if speeds[i] >= p.max_speed:
                continue
            tips_in = int(_in_region(ff.to_uv(h.lm[_TIPS]), p.half_width, v_lo, v_hi).sum())
            pinch_pt = ff.to_uv(((h.lm[THUMB_TIP, :2] + h.lm[INDEX_TIP, :2]) / 2.0)[None])[0]
            pinched = dist(h.lm[THUMB_TIP], h.lm[INDEX_TIP]) / max(h.size, 1e-6) < p.pinch
            if tips_in >= p.min_tips or (pinched and bool(_in_region(pinch_pt, p.half_width, v_lo, v_hi))):
                return True
        return False

    def _hand_signs(self, hands: list[Hand], fw: float, now: float) -> dict[str, bool]:
        """One-hand signs: the classifier's label, backed by the finger shape, on a still hand off the face."""
        out = dict.fromkeys(HAND_SIGNS, False)
        why: dict[str, str] = {}  # a sign was shown but didn't count: tell the user why (HUD)
        self.debug["sign_why"] = why
        box = self._box if now - self._box_seen <= self.params[COVER_FACE].bbox_memory_s else None
        for i, h in enumerate(hands):
            for name in HAND_SIGNS:
                p: HandSignParams = self.params[name]
                labelled = h.gesture == SIGN_LABELS[name] and h.gesture_score >= p.min_score
                if not labelled and not (p.shape_only and h.gesture in ("None", SIGN_LABELS[name])):
                    continue
                shape = SIGN_SHAPES[name](h)
                on_face = box is not None and (
                    overlap_ratio(hand_bbox(h), box) > p.max_face_overlap
                    or any(_inside(h.lm[k, :2], box) for k in SIGN_TIPS[name])
                )
                speed = self._hands.speed(i, now, 0.25) / fw  # windowed: single frames jitter
                self.debug[name] = {"score": round(h.gesture_score, 2), "shape": shape, "on_face": on_face,
                                    "speed": round(speed, 2)}
                sure = labelled and h.gesture_score >= p.sure_score
                if not (shape or sure):
                    if labelled:
                        why.setdefault(name, "fingers?")
                elif on_face:
                    why.setdefault(name, "on your face")
                elif speed >= p.max_speed:
                    why.setdefault(name, "hold still")
                else:
                    out[name] = True
        for name in HAND_SIGNS:
            if out[name]:
                why.pop(name, None)  # the other hand shows it properly
        return out

    def _hands_on_face(self, hands: list[Hand], sig: Signals, now: float) -> int:
        """How many hands cover the face bbox (or where the face was, while it can't be seen).

        The palm centre must be over the face too: hands raised beside the head (Absolute
        Cinema) overlap the bbox a lot, and dropping them must not read as an uncovering.
        """
        p: CoverFaceParams = self.params[COVER_FACE]
        if sig.face is not None:
            self._box, self._box_seen = sig.face.bbox(), now
            stale = not sig.face.fresh
        elif self._box is not None and now - self._box_seen <= p.bbox_memory_s:
            stale = True
        else:
            return 0
        x0, y0, x1, y1 = self._box
        m = p.palm_margin * (x1 - x0)
        grown = (x0 - m, y0 - m, x1 + m, y1 + m)
        return sum(
            1
            for h in hands
            if (overlap_ratio(hand_bbox(h), self._box) >= p.min_overlap and _inside(h.palm_center, grown))
            or (stale and _inside(h.palm_center, self._box))
        )

    def _eyebrows(self, sig: Signals, now: float) -> bool:
        p: EyebrowsParams = self.params[EYEBROWS_UP]
        face = sig.face
        near = any(face is None or float(np.hypot(*(h.palm_center - face.center))) < p.hand_near_fw * face.width
                   for h in sig.hands)
        if near or self._occl is not None:  # a hand at the face (or lost on the forehead) blocks it
            self._hand_seen = now
        hands_clear = now - self._hand_seen > p.hand_clear_s
        if sig.face is not None and sig.face.fresh:
            self._brows.update(sig.face, now, hands_clear)
        else:
            self._brows.expire(now)
        bw = self._brows
        fp: ForeheadParams = self.params[FOREHEAD]
        pained = bw.ema.get("browInnerUp") >= fp.fast_brow_inner or (
            0.5 * (bw.ema.get("mouthFrownLeft") + bw.ema.get("mouthFrownRight")) >= fp.fast_frown
        )
        self.t_on[FOREHEAD] = fp.t_on_fast if pained else fp.t_on
        self.t_on[EYEBROWS_UP] = p.t_on * (p.speak_factor if bw.speaking else 1.0)
        self.debug[EYEBROWS_UP] = {"level": round(bw.level, 2), "base": round(bw.base, 2),
                                   "threshold": round(bw.threshold, 2), "asym": round(bw.asym, 2),
                                   "speaking": bw.speaking, "hands_clear": hands_clear}
        return bw.raised and hands_clear

    def _cancel(self, h: Hand) -> bool:
        p: CancelParams = self.params[CANCEL]
        return int(fingers_extended(h).sum()) >= p.min_fingers or (
            h.gesture == "Open_Palm" and h.gesture_score >= p.open_palm
        )

    def _thumb_up_any(self, h: Hand) -> bool:
        """Thumbs up by MediaPipe's classifier or by shape; hands low in the frame count too."""
        p = self.params[TWO_THUMBS]
        if h.score < p.min_hand_score:
            return False
        return (h.gesture == "Thumb_Up" and h.gesture_score >= p.min_score) or thumb_up_shape(h)

    @property
    def brow_values(self) -> tuple[float, float]:
        """(face-model brow level, brow height over the eyelids) of the last fresh face."""
        return self._brows.level, self._brows.geo

    def set_brow_calibration(self, cal: dict | None) -> None:
        """{"signal": "blend" | "geo", "rise": x} from Ctrl+Alt+C, or None for the defaults."""
        b = self._brows
        if cal and cal.get("signal") in ("blend", "geo") and isinstance(cal.get("rise"), (int, float)):
            b.signal, b.rise = cal["signal"], float(cal["rise"])
        else:
            b.signal, b.rise = "blend", None
        b._recent.clear()

    def _thumbs_up(self, h: Hand) -> bool:
        return h.gesture == "Thumb_Up" and h.gesture_score >= self.params[THUMBS_UP].min_score

    def _loaded(self, sig: Signals, fw: float, g: GeneralParams, ff: FaceFrame | None,
                lv: _Levels | None) -> list[np.ndarray]:
        """Loaded snaps, ignoring hands on the face (a forehead pinch can look like a loaded snap)
        and at the collar (fingers pinching a collar look loaded too; seen in real recordings)."""
        loaded = snap_loaded_hands(sig, g.snap_pinch, g.snap_reach, g.min_score)
        out = []
        cp: CollarParams = self.params[COLLAR]
        for pc in loaded:
            if self._box is not None:
                x0, y0, x1, y1 = self._box
                m = 0.25 * fw
                if _inside(pc, (x0 - m, y0 - m, x1 + m, y1 + m)):
                    continue
            if ff is not None and lv is not None:
                u, v = ff.to_uv(pc[None])[0]
                if abs(u) <= cp.half_width + 0.15 and lv.chin - 0.35 <= v <= lv.chin + cp.bottom:
                    continue
            out.append(pc)
        return out
