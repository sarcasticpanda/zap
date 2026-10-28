"""Snap detection (seen by the camera, optionally heard by the mic) and burst counting.

Default (``vision_only``): a snap is SEEN. The hand is "loaded" (thumb tip on
the middle fingertip, middle finger not curled) and, within ``V_RELEASE_S``,
the middle finger flicks down toward the palm while the thumb springs away.
Noise in the room can't fake that, and it works while you are invisible (the
camera still sees you).

Mic mode: the microphone knows WHEN a snap happened (to the millisecond); a
30 fps camera only sees the hand before and after. So an accepted audio
candidate becomes a snap only if vision AGREES: around the onset some tracked hand was "loaded"
(thumb pressed on the middle finger, middle finger extended), released, or
vanished mid-snap. Sounds with no usable hand in view - including snaps played
by other meeting participants through the speakers - are dropped, and two
palms meeting is a clap.

Hand features use RAW landmarks: smoothing would blur the 1-frame release.
With s = |P0 - P9| (wrist to middle MCP):
  tm = |P4 - P12| / s   thumb tip to middle tip (small when loaded)
  mw = |P12 - P0| / s   middle tip to wrist (large while the middle finger is up)

``BurstCounter`` groups accepted onsets into 1-3 snap bursts. Typical use, once
per frame::

    for c in mic.drain():
        fusion.add_candidate(c)
    fusion.add_frame(sig)
    for t, where in fusion.resolve(now):
        marker = bursts.add(t, where)          # Intent("charge") or None
    done, markers = bursts.tick(now, mic.pending_onset, fusion.pending_onset)
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from zap.gestures.poses import thumb_up_shape
from zap.gestures.snap_audio import AudioCandidate
from zap.gestures.types import Intent, SnapBurst
from zap.signals import MIDDLE_MCP, MIDDLE_TIP, THUMB_TIP, WRIST, Hand, Signals

log = logging.getLogger(__name__)

Point = tuple[float, float]

# Pose thresholds (ratios of the hand scale s).
LOAD_TM, LOAD_MW, LOAD_FRAMES = 0.31, 1.11, 2
RELEASE_S = 0.375  # a release must follow the last loaded frame within this
RELEASE_TM, RELEASE_DROP = 0.44, 0.18
# Vision-only snaps (the default). Loose enough for a 20-30 fps camera that
# often sees only one loaded frame, then a blurred or missing hand, then the
# finished snap; the flick distance keeps a slow pinch-and-open from counting.
# Measured on the user's snaps: loaded tm 0.03-0.18 / mw 1.29-1.51; just after the
# snap tm 0.52-0.64 / mw 0.74-0.80 (middle finger curled into the palm).
# A collar pinch being let go also separates thumb and middle finger, but the middle
# finger never curls into the palm (mw stayed >= 1.1), and it uses both hands.
V_LOAD_TM, V_LOAD_MW = 0.38, 1.0
V_RELEASE_S = 0.45  # the finished snap must be seen within this after the last loaded frame
V_REL_TM = 0.38  # thumb and middle fingertip apart again ...
V_REL_MW = 0.95  # ... the middle finger curled into the palm ...
V_REL_DROP = 0.25  # ... having come down at least this far ...
V_REL_MOVE = 0.30  # ... and the middle tip travelled at least this far (hand sizes)
V_TWO_HANDS_S = 0.5  # another hand loaded this recently: a two-hand gesture, not a snap
V_REFRACTORY_S = 0.4  # landmark jitter right after a snap must not count as a second snap
# After a real snap the index finger stays out (measured 0.93-1.13); making a thumbs-up, a fist or
# letting go of a collar curls it (~0.78). |index tip - wrist| / |index PIP - wrist|:
V_INDEX_EXT = 0.87
V_AFTER_THUMB_S = 1.0  # no snaps from a hand that showed a thumbs-up this recently (money gesture)
V_MAX_WRIST_Y = 1.1  # close to the camera the wrist may sit below the frame; the fingers still count
V_SENS = {1: 0.8, 2: 0.9, 3: 1.0, 4: 1.12, 5: 1.25}  # snap.sensitivity -> looseness
CLAP_DIST = 1.2  # palms closer than this * s

# Usable hand.
MIN_SCORE, MIN_SIZE, MAX_WRIST_Y = 0.6, 0.045, 0.92

# Fusion windows around tv = t_onset + av_offset (seconds).
LOADED_WIN = (-0.45, 0.05)
RELEASE_WIN = (-0.10, 0.25)
LOST_WIN = (-0.05, 0.15)
CLAP_WIN = (-0.15, 0.10)
SEARCH_WIN = (-0.45, 0.25)  # any usable hand / same-hand-in-burst

TRACK_GATE = 2.0  # max palm jump between frames, in hand sizes ...
TRACK_GATE_W = 0.08  # ... but at least this fraction of the frame width
TRACK_KEEP_S = 0.3  # a missing track can be re-associated for this long
HISTORY_S = 2.5

# Fusion reject reasons (HUD).
NO_HAND = "no hand"
CLAP = "clap"
NO_POSE = "no snap pose"


def snap_features(hand: Hand) -> tuple[float, float, float]:
    """(s, tm, mw) from raw landmarks; tm/mw are inf for a degenerate hand."""
    p = hand.lm[:, :2]
    s = float(np.hypot(*(p[WRIST] - p[MIDDLE_MCP])))
    if s < 1e-6:
        return s, math.inf, math.inf
    tm = float(np.hypot(*(p[THUMB_TIP] - p[MIDDLE_TIP]))) / s
    mw = float(np.hypot(*(p[MIDDLE_TIP] - p[WRIST]))) / s
    return s, tm, mw


def is_usable(hand: Hand, s: float, width: int, height: int) -> bool:
    return hand.score >= MIN_SCORE and s >= MIN_SIZE * width and float(hand.lm[WRIST, 1]) <= MAX_WRIST_Y * height


@dataclass
class _Track:
    tid: int
    pos: np.ndarray
    s: float
    t: float  # last frame the hand was seen
    streak: int = 0  # consecutive loaded frames
    load_t: float = -math.inf  # last loaded frame
    load_mw: float = 0.0
    load_tip: tuple[float, float] = (0.0, 0.0)  # middle fingertip in the last loaded frame
    seen_loaded_t: float = -math.inf  # last loaded frame, kept after a snap (two-hand check)
    index_ext: float = 1.0  # index finger extension in the latest frame
    thumb_t: float = -math.inf  # last frame this hand showed a thumbs-up
    snap_t: float = -math.inf  # this hand's last snap
    missing: bool = False


@dataclass
class _Frame:
    t: float
    usable: dict[int, tuple[float, float, float]] = field(default_factory=dict)  # tid -> (x, y, s)
    loaded: dict[int, Point] = field(default_factory=dict)  # tid -> palm centre
    clap: bool = False


@dataclass
class _Event:
    t: float
    tid: int
    where: Point


class SnapFusion:
    """Confirms accepted audio candidates with hand poses seen by the camera.

    ``add_frame(sig)`` every perception frame; ``add_candidate(c)`` for audio
    candidates (non-accepted ones are ignored); ``resolve(now)`` returns the
    confirmed snaps as ``(t_onset, where)``, oldest first, once enough video
    after the onset has arrived (or a timeout passed). ``rejects`` holds the
    last 5 fusion reject reasons for the HUD.

    ``vision_only`` (the default; also set at runtime when the mic fails):
    audio candidates are ignored and a seen load -> flick produces onsets
    directly, so bursts of 1-3 snaps still count.
    """

    def __init__(
        self,
        av_offset_s: float = 0.08,
        vision_only: bool = False,
        burst_gap_s: float = 0.65,
        cover_s: float = 0.25,
        timeout_s: float = 0.30,
    ) -> None:
        self.av_offset_s = av_offset_s
        self.vision_only = vision_only
        self.sensitivity = 3
        self.burst_gap_s = burst_gap_s
        self.cover_s = cover_s
        self.timeout_s = timeout_s
        self.rejects: deque[str] = deque(maxlen=5)
        self.trace = None  # optional text file: one JSON line of hand features per frame (tuning)
        self.reset()

    def reset(self) -> None:
        self._frames: deque[_Frame] = deque()
        self._releases: deque[_Event] = deque()
        self._losts: deque[_Event] = deque()
        self._tracks: list[_Track] = []
        self._next_tid = 0
        self._pending: list[float] = []
        self._vision_onsets: list[tuple[float, Point | None]] = []
        self._latest_t = -math.inf
        self._last_accept_t = -math.inf
        self._last_tid: int | None = None
        self._last_where: Point | None = None
        self.loaded_now = False  # some hand is loaded in the latest frame (HUD)
        self.hand_up = False  # some usable hand in the latest frame (a burst may continue)

    @property
    def watching(self) -> bool:
        """A loaded hand may snap any moment: run the hand model on every frame."""
        return any(self._latest_t - tr.load_t <= V_RELEASE_S for tr in self._tracks)

    @property
    def pending_onset(self) -> float | None:
        """Earliest onset still waiting for video, if any (hold for BurstCounter)."""
        return min(self._pending) if self._pending else None

    # ------------------------------------------------------------ inputs
    def add_candidate(self, c: AudioCandidate) -> None:
        if c.accepted and not self.vision_only:
            self._pending.append(c.t_onset)

    def add_frame(self, sig: Signals) -> None:
        t = sig.t
        self._latest_t = max(self._latest_t, t)
        feats = [snap_features(h) for h in sig.hands]
        centers = [h.palm_center.astype(np.float64) for h in sig.hands]
        tracks = self._associate(centers, feats, t, sig.width)
        frame = _Frame(t, clap=_palms_touch(centers, feats))
        for hand, (s, tm, mw), c, tr in zip(sig.hands, feats, centers, tracks, strict=True):
            where = (float(c[0]), float(c[1]))
            usable = is_usable(hand, s, sig.width, sig.height) if not self.vision_only else (
                hand.score >= MIN_SCORE and s >= MIN_SIZE * sig.width
                and float(hand.lm[WRIST, 1]) <= V_MAX_WRIST_Y * sig.height)
            if usable:
                frame.usable[tr.tid] = (where[0], where[1], s)
            tip = (float(hand.lm[MIDDLE_TIP, 0]), float(hand.lm[MIDDLE_TIP, 1]))
            p = hand.lm[:, :2]
            tr.index_ext = float(np.hypot(*(p[8] - p[WRIST]))) / max(1e-6, float(np.hypot(*(p[6] - p[WRIST]))))
            if hand.gesture == "Thumb_Up" or thumb_up_shape(hand):
                tr.thumb_t = t
            if self.trace is not None:
                self.trace.write(json.dumps({"t": round(t, 3), "id": tr.tid, "s": round(s, 1), "tm": round(tm, 3),
                                             "mw": round(mw, 3), "tip": [round(tip[0]), round(tip[1])],
                                             "ok": usable, "g": hand.gesture, "gs": round(hand.gesture_score, 2),
                                             "lm": hand.lm[:, :2].round(1).tolist()}) + "\n")
            self._update_pose(tr, frame, where, usable, tm, mw, t, tip)
        self.loaded_now = bool(frame.loaded)
        self.hand_up = bool(frame.usable)
        self._frames.append(frame)
        self._prune(t)

    def _associate(
        self, centers: list[np.ndarray], feats: list[tuple[float, float, float]], t: float, width: int
    ) -> list[_Track]:
        """Match hands to tracks by nearest palm centre (greedy); new tracks for the rest."""
        live = [tr for tr in self._tracks if t - tr.t <= TRACK_KEEP_S]
        pairs = sorted(
            (float(np.hypot(*(c - tr.pos))), i, j) for i, c in enumerate(centers) for j, tr in enumerate(live)
        )
        out: list[_Track | None] = [None] * len(centers)
        used: set[int] = set()
        for d, i, j in pairs:
            if out[i] is not None or j in used:
                continue
            if d <= max(TRACK_GATE * max(live[j].s, feats[i][0]), TRACK_GATE_W * width):
                out[i], used = live[j], used | {j}
        for j, tr in enumerate(live):
            if j not in used and not tr.missing:
                tr.missing = True
                where = (float(tr.pos[0]), float(tr.pos[1]))
                if not self.vision_only and t - tr.load_t <= RELEASE_S:  # vanished mid-snap: motion blur
                    self._losts.append(_Event(t, tr.tid, where))
                tr.streak = 0
        for i, c in enumerate(centers):
            if out[i] is None:
                out[i] = _Track(self._next_tid, c, feats[i][0], t)
                self._next_tid += 1
                live.append(out[i])
            tr = out[i]
            tr.pos, tr.s, tr.t, tr.missing = c, feats[i][0], t, False
        self._tracks = live
        return out  # type: ignore[return-value]

    def _update_pose(
        self, tr: _Track, frame: _Frame, where: Point, usable: bool, tm: float, mw: float, t: float,
        tip: Point = (0.0, 0.0),
    ) -> None:
        if self.vision_only:
            self._update_vision(tr, frame, where, usable, tm, mw, t, tip)
            return
        if usable and tm < LOAD_TM and mw > LOAD_MW:
            tr.streak += 1
            if tr.streak >= LOAD_FRAMES:
                tr.load_t, tr.load_mw = t, mw
                frame.loaded[tr.tid] = where
            return
        tr.streak = 0
        if t - tr.load_t > RELEASE_S:
            return
        if tm > RELEASE_TM and mw <= tr.load_mw - RELEASE_DROP:
            self._releases.append(_Event(t, tr.tid, where))
            tr.load_t = -math.inf

    def _update_vision(
        self, tr: _Track, frame: _Frame, where: Point, usable: bool, tm: float, mw: float, t: float, tip: Point
    ) -> None:
        k = V_SENS.get(self.sensitivity, 1.0)
        if usable and tm < V_LOAD_TM * k and mw > V_LOAD_MW / k and t - tr.snap_t > V_REFRACTORY_S:
            tr.load_t, tr.load_mw, tr.load_tip, tr.seen_loaded_t = t, mw, tip, t
            frame.loaded[tr.tid] = where
            return
        if not usable or t - tr.load_t > V_RELEASE_S:
            return
        travel = math.hypot(tip[0] - tr.load_tip[0], tip[1] - tr.load_tip[1]) / max(tr.s, 1e-6)
        curled = mw <= min(V_REL_MW * k, tr.load_mw - V_REL_DROP / k)
        if tm > V_REL_TM / k and curled and travel >= V_REL_MOVE / k:
            tr.load_t = -math.inf
            if self._other_loaded(tr):
                self._log_snap(t, tr.tid, "two hands")
                return
            if tr.index_ext < V_INDEX_EXT:  # a thumbs-up or a fist being made, not a snap
                self._log_snap(t, tr.tid, "index curled")
                return
            if t - tr.thumb_t <= V_AFTER_THUMB_S:  # the hand coming out of a thumbs-up
                self._log_snap(t, tr.tid, "after thumbs-up")
                return
            # The snap happened between the last loaded frame and this one.
            tr.snap_t = t
            self._vision_onsets.append((0.5 * (tr.seen_loaded_t + t), where))
            self._log_snap(t, tr.tid, "seen")

    def _other_loaded(self, tr: _Track) -> bool:
        return any(o is not tr and tr.t - o.seen_loaded_t <= V_TWO_HANDS_S for o in self._tracks)

    def _log_snap(self, t: float, tid: int, how: str) -> None:
        if self.trace is not None:
            self.trace.write(json.dumps({"t": round(t, 3), "snap": tid, "how": how}) + "\n")

    def _prune(self, t: float) -> None:
        horizon = t - HISTORY_S
        for q in (self._frames, self._releases, self._losts):
            while q and q[0].t < horizon:
                q.popleft()

    # ------------------------------------------------------------ decisions
    def resolve(self, now: float, in_burst: bool | None = None) -> list[tuple[float, Point | None]]:
        """Confirmed snap onsets, oldest first.

        ``in_burst`` hints that a burst is open (``BurstCounter.in_burst``), so
        a candidate is also accepted when the hand of the previous snap is still
        in view. When None it is inferred from this object's last accepted onset.
        """
        out = self._vision_onsets
        self._vision_onsets = []
        if self.vision_only:
            self._pending.clear()
            return out
        keep = []
        for t_onset in sorted(self._pending):
            tv = t_onset + self.av_offset_s
            if self._latest_t < tv + self.cover_s and now < tv + self.cover_s + self.timeout_s:
                keep.append(t_onset)
                continue
            burst = in_burst if in_burst is not None else t_onset - self._last_accept_t <= self.burst_gap_s
            ok, info = self._decide(tv, burst)
            if ok:
                where, tid = info
                self._last_accept_t, self._last_tid, self._last_where = t_onset, tid, where
                out.append((t_onset, where))
            else:
                self.rejects.append(info)
                log.debug("snap at %.3f rejected: %s", t_onset, info)
        self._pending = keep
        return out

    def _decide(self, tv: float, in_burst: bool):
        """(True, (where, tid)) or (False, reason)."""
        frames = _window(self._frames, tv, SEARCH_WIN)
        if not any(f.usable for f in frames):
            return False, NO_HAND
        if any(f.clap for f in _window(self._frames, tv, CLAP_WIN)):
            return False, CLAP
        rel = _window(self._releases, tv, RELEASE_WIN)
        if rel:
            return True, (rel[0].where, rel[0].tid)
        for f in reversed(_window(self._frames, tv, LOADED_WIN)):
            if f.loaded:
                tid, where = next(iter(f.loaded.items()))
                return True, (where, tid)
        lost = _window(self._losts, tv, LOST_WIN)
        if lost:
            return True, (lost[0].where, lost[0].tid)
        if in_burst:
            hit = self._same_hand(frames)
            if hit is not None:
                return True, hit
        return False, NO_POSE

    def _same_hand(self, frames: list[_Frame]) -> tuple[Point, int] | None:
        """The previous snap's hand (same track, or a usable hand near its position)."""
        last = self._last_where
        for f in reversed(frames):
            for tid, (x, y, s) in f.usable.items():
                near = last is not None and math.hypot(x - last[0], y - last[1]) <= TRACK_GATE * s
                if tid == self._last_tid or near:
                    return (x, y), tid
        return None


def _window(items, tv: float, win: tuple[float, float]) -> list:
    lo, hi = tv + win[0], tv + win[1]
    return [it for it in items if lo <= it.t <= hi]


def _palms_touch(centers: list[np.ndarray], feats: list[tuple[float, float, float]]) -> bool:
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if float(np.hypot(*(centers[i] - centers[j]))) < CLAP_DIST * max(feats[i][0], feats[j][0]):
                return True
    return False


class BurstCounter:
    """Groups snap onsets into bursts of 1..max_count.

    ``add(t, where) -> Intent | None``: returns ``Intent("charge", t=t, where=where)``
    when ``t`` opens a new burst (start the ember pre-roll); None otherwise
    (joined the open burst, merged within ``refractory_s``, or ignored during the
    lockout that follows a dispatch).

    ``tick(now, *pending) -> (bursts, markers)``: ``bursts`` lists completed
    ``SnapBurst`` objects (a burst completes when ``now - last onset > gap_s``);
    ``markers`` lists ``Intent("discharge")`` when a burst of more than
    ``max_count`` snaps is discarded. ``pending`` are onset times still being
    judged upstream (``MicStream.pending_onset``, ``SnapFusion.pending_onset``;
    None entries are ignored): while one lies within the gap of the open burst,
    the burst is held open. With ``clamp=True`` an oversized burst dispatches
    as ``max_count`` instead (vision-only mode: ``max_count=1, clamp=True``).
    """

    def __init__(
        self,
        gap_s: float = 0.65,
        refractory_s: float = 0.15,
        lockout_s: float = 0.8,
        max_count: int = 3,
        clamp: bool = False,
        quiet_s: float | None = None,
    ) -> None:
        self.gap_s = gap_s
        self.quiet_s = gap_s if quiet_s is None else min(quiet_s, gap_s)  # close early unless held
        self.refractory_s = refractory_s
        self.lockout_s = lockout_s
        self.max_count = max_count
        self.clamp = clamp
        self._onsets: list[float] = []
        self._where: Point | None = None
        self._lock_until = -math.inf

    @property
    def in_burst(self) -> bool:
        return bool(self._onsets)

    @property
    def count(self) -> int:
        return len(self._onsets)

    def reset(self) -> None:
        self._onsets.clear()
        self._where = None
        self._lock_until = -math.inf

    def add(self, t: float, where: Point | None = None) -> Intent | None:
        if t < self._lock_until:
            return None
        if self._onsets:
            if min(abs(t - o) for o in self._onsets) < self.refractory_s:
                return None
            if self._onsets[0] - self.gap_s <= t <= self._onsets[-1] + self.gap_s:
                self._onsets.append(t)
                self._onsets.sort()
                self._where = where if where is not None else self._where
                return None
            # Too late for the open burst (tick was not called in time): the
            # open burst stays for tick() to complete; t is dropped.
            log.debug("snap onset %.3f outside the open burst, dropped", t)
            return None
        self._onsets.append(t)
        self._where = where
        return Intent("charge", t=t, where=where)

    def tick(self, now: float, *pending: float | None, hold: bool = True) -> tuple[list[SnapBurst], list[Intent]]:
        """``hold=False`` (e.g. the snapping hand went down) completes the burst after ``quiet_s``."""
        if not self._onsets:
            return [], []
        last = self._onsets[-1]
        full = len(self._onsets) >= self.max_count  # no bigger burst can come: done now
        if not full and now - last <= (self.gap_s if hold else self.quiet_s):
            return [], []
        if not full and any(p is not None and p <= last + self.gap_s for p in pending):
            return [], []
        n, first, where = len(self._onsets), self._onsets[0], self._where
        self._onsets.clear()
        self._where = None
        if n > self.max_count and not self.clamp:
            log.debug("snap burst of %d discarded", n)
            return [], [Intent("discharge", t=now, where=where)]
        self._lock_until = now + self.lockout_s
        return [SnapBurst(min(n, self.max_count), first, last, where)], []
