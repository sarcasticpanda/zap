"""Ctrl+Alt+C: calibrate every gesture to the person in front of the camera, in about a minute.

The preview walks you through short steps: sit relaxed, talk, look down and
back, raise your eyebrows twice, put a hand on your forehead, touch your face
naturally, then show each gesture once. What the camera saw (landmarks only,
kept in memory, never written to disk) is then replayed through the real
:class:`PoseDetector` with candidate settings. For each tunable gesture the
first candidate (in order of preference) that fires in its own step and never
anywhere else is kept, so ordinary talking, glasses, head tilts and touching
your face don't set off Jim or Michael Scott. The other gestures are checked
and reported.

The analysis is a pure function of the recorded frames (:func:`analyse`), so it
also runs on ``zap record`` recordings and in tests.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from zap.gestures.poses import PoseDetector
from zap.gestures.types import (
    COLLAR,
    COVER_FACE,
    EYEBROWS_UP,
    FOREHEAD,
    HANDS_UP,
    L_SHAPE,
    PEACE,
    TWO_THUMBS,
)
from zap.signals import Signals

log = logging.getLogger(__name__)

VERSION = 2
REACT_S = 1.0  # each step starts with this long to react to the prompt: not analysed
LATE_S = 2.0  # ... and the step after a gesture's own step this long (people let go of a pose late)
LATE_POS_S = 1.0  # a gesture shown this late (into the next step) still counts for its own step
ANALYSE_FPS = 15.0
YIELD_EVERY = 25  # frames between short sleeps while replaying


@dataclass(frozen=True)
class Step:
    name: str
    seconds: float
    prompt: str


STEPS: tuple[Step, ...] = (
    Step("relax", 4.0, "look at the camera, face relaxed"),
    Step("talk", 6.0, "talk normally: say anything, smile, look around"),
    Step("look_down", 4.0, "look down at your keyboard, then back at the camera"),
    Step("raise1", 3.5, "RAISE your eyebrows high and hold"),
    Step("rest", 2.5, "relax your eyebrows"),
    Step("raise2", 3.5, "RAISE your eyebrows again and hold"),
    Step("forehead", 4.5, "hand flat on your forehead (Michael Scott) and hold"),
    Step("touch", 5.0, "touch your face naturally: rub your eyes, fix your glasses or hair, scratch"),
    Step("peace", 3.5, "PEACE sign beside your face, hold"),
    Step("l_shape", 3.5, "L sign beside your face (index up, thumb out), hold"),
    Step("thumbs", 3.0, "THUMBS UP, hold"),
    Step("hands_up", 3.5, "BOTH open hands up beside your head, hold"),
    Step("collar", 3.5, "fingers at your COLLAR, hold"),
    Step("cover", 4.5, "COVER your face with both hands, then drop them"),
)

# which steps each gesture should fire in (everywhere else it must not)
POSITIVE: dict[str, tuple[str, ...]] = {
    EYEBROWS_UP: ("raise1", "raise2"),
    FOREHEAD: ("forehead",),
    PEACE: ("peace",),
    L_SHAPE: ("l_shape",),
    TWO_THUMBS: ("thumbs",),
    HANDS_UP: ("hands_up",),
    COLLAR: ("collar",),
    COVER_FACE: ("cover",),
}
EVENTS = (COVER_FACE,)  # one-frame truths: fired = seen at least once in the step
NAMES = {EYEBROWS_UP: "Jim (eyebrows)", FOREHEAD: "Michael Scott (forehead)", PEACE: "peace sign",
         L_SHAPE: "L sign", TWO_THUMBS: "thumbs up", HANDS_UP: "both hands up", COLLAR: "collar",
         COVER_FACE: "cover face"}
HINTS = {
    EYEBROWS_UP: "raise them higher, looking straight at the camera",
    FOREHEAD: "lay your hand flat across your forehead and keep it still",
    PEACE: "hold it up beside your face, palm to the camera",
    L_SHAPE: "index finger straight up, thumb straight out, other fingers folded",
    TWO_THUMBS: "thumb straight up, fist closed",
    HANDS_UP: "both open palms up beside your head, not above it",
    COLLAR: "pinch your collar just under your chin",
    COVER_FACE: "cover your whole face for a second, then drop both hands",
}
FIRE_RUN_S = {EYEBROWS_UP: 0.6, FOREHEAD: 0.7, PEACE: 0.7, L_SHAPE: 0.7, TWO_THUMBS: 0.7, HANDS_UP: 0.4,
              COLLAR: 0.6}  # a held pose must be seen this long without a gap to fire
MAX_GAP_S = 0.15  # gaps shorter than this don't break a run (the hand model drops frames)
FALSE_RUN_SHARE = 0.5  # elsewhere, runs must stay under this share of the firing time


# ---------------------------------------------------------------- recording the steps
@dataclass
class Frame:
    step: str
    t: float  # seconds into the step
    sig: Signals


class CalibrationRun:
    """Drives the steps from the frame loop: ``feed`` every frame until it returns None."""

    def __init__(self, now: float, steps: Sequence[Step] = STEPS) -> None:
        self.steps = tuple(steps)
        self.i = 0
        self.t0 = now
        self.frames: list[Frame] = []

    @property
    def done(self) -> bool:
        return self.i >= len(self.steps)

    def feed(self, sig: Signals, now: float) -> tuple[int, Step, float] | None:
        """Record this frame; returns (step number, step, seconds left) or None once all steps are done."""
        while not self.done and now - self.t0 >= self.steps[self.i].seconds:
            self.t0 += self.steps[self.i].seconds
            self.i += 1
        if self.done:
            return None
        step = self.steps[self.i]
        lean = Signals(sig.t, sig.seq, list(sig.hands), sig.face, None, False, sig.width, sig.height)
        self.frames.append(Frame(step.name, now - self.t0, lean))
        return self.i + 1, step, step.seconds - (now - self.t0)


# ---------------------------------------------------------------- analysis
@dataclass
class Verdict:
    gesture: str
    fired: bool  # fired in its own step(s)
    false_s: float  # longest stretch it was on anywhere else, seconds
    params: dict[str, Any] = field(default_factory=dict)  # the chosen tuning (detector side)
    note: str = ""
    margin: float = FALSE_RUN_SHARE  # tuned gestures keep a safety margin; checked ones just mustn't fire

    @property
    def ok(self) -> bool:
        return self.fired and self.false_s < self.margin * FIRE_RUN_S.get(self.gesture, 0.6)

    def line(self) -> str:
        name = NAMES.get(self.gesture, self.gesture)
        if self.ok:
            return f"{name}: OK" + (f" ({self.note})" if self.note else "")
        if not self.fired:
            return f"{name}: NOT SEEN - {HINTS.get(self.gesture, 'try again')}"
        if self.gesture == EYEBROWS_UP:
            return f"{name}: your brows go that high while you talk too - raise them higher ({self.note})"
        return f"{name}: also fired when you weren't doing it ({self.note or 'kept the strictest setting'})"


@dataclass
class Result:
    verdicts: dict[str, Verdict]
    brows: dict[str, Any] | None
    params: dict[str, dict[str, Any]]

    def calibration(self) -> dict[str, Any]:
        return {"version": VERSION, "brows": self.brows, "params": self.params,
                "report": {g: v.line() for g, v in self.verdicts.items()}, "when": time.strftime("%Y-%m-%d %H:%M")}


def _runs(frames: Sequence[Frame], on: Sequence[bool], counted: Sequence[bool]) -> float:
    """Longest run (seconds) of True among the counted frames (an uncounted frame ends a run)."""
    best = 0.0
    start = last = None
    for f, v, c in zip(frames, on, counted, strict=True):
        if not c:
            start = last = None
            continue
        t = f.sig.t
        if v:
            if start is None or (last is not None and t - last > MAX_GAP_S):
                start = t
            last = t
            best = max(best, last - start)
        elif last is not None and t - last > MAX_GAP_S:
            start = last = None
    return best


def _evaluate(frames: Sequence[Frame], truths: Sequence[Mapping[str, bool]], gesture: str,
              positive: Mapping[str, Sequence[str]]) -> tuple[bool, float]:
    """(fired in each of its own steps, longest stretch it was on anywhere else)."""
    on = [bool(tr.get(gesture, False)) for tr in truths]
    windows = _windows(frames, positive.get(gesture, ()))
    if not windows:
        return False, 0.0
    late = _late(frames, positive.get(gesture, ()))
    pos = set(positive.get(gesture, ()))
    elsewhere = [f.step not in pos and f.t >= (LATE_S if late[i] else REACT_S) for i, f in enumerate(frames)]
    if gesture in EVENTS:  # it happens once (e.g. as the hands come off the face), in any of its steps
        fired = any(v and w for win in windows for v, w in zip(on, win, strict=True))
        return fired, (1.0 if any(v and e for v, e in zip(on, elsewhere, strict=True)) else 0.0)
    need = FIRE_RUN_S.get(gesture, 0.6)
    fired = all(_runs(frames, on, win) + 1e-6 >= need for win in windows)
    return fired, _runs(frames, on, elsewhere)


def _windows(frames: Sequence[Frame], steps: Sequence[str]) -> list[list[bool]]:
    """Per positive step, the frames where the gesture is expected: the step after the reaction time,
    and the first LATE_POS_S of the next step (people often get there a moment late)."""
    order = list(dict.fromkeys(f.step for f in frames))
    out = []
    for s in (s for s in order if s in steps):
        nxt = order[order.index(s) + 1] if order.index(s) + 1 < len(order) else None
        out.append([(f.step == s and f.t >= REACT_S) or (f.step == nxt and f.t < LATE_POS_S) for f in frames])
    return out


def _late(frames: Sequence[Frame], steps: Sequence[str]) -> list[bool]:
    """Frames in the step right after one of ``steps`` (where letting go late is forgiven up to LATE_S)."""
    order = list(dict.fromkeys(f.step for f in frames))
    after = {order[i + 1] for i, s in enumerate(order[:-1]) if s in steps}
    return [f.step in after for f in frames]


def _replay(frames: Sequence[Frame], overrides: Mapping[str, Mapping[str, Any]] | None,
            cal: Mapping[str, Any] | None) -> tuple[list[dict[str, bool]], list[tuple[int, float, float]]]:
    """Run the frames through a fresh detector; returns per-frame truths and (frame index, brow level,
    brow height) for fresh faces. Sleeps a moment now and then: on a background thread this keeps the
    video loop smooth."""
    det = PoseDetector(overrides)
    det.apply_calibration(dict(cal) if cal else None)
    truths, brows = [], []
    for i, f in enumerate(frames):
        truths.append(det.update(f.sig, f.sig.t))
        if f.sig.face is not None and f.sig.face.fresh:
            lvl, geo = det.brow_values
            brows.append((i, lvl, geo))
        if i % YIELD_EVERY == YIELD_EVERY - 1:
            time.sleep(0.001)
    return truths, brows


def _thin(frames: Sequence[Frame], fps: float = ANALYSE_FPS) -> list[Frame]:
    """Every gesture rule is timed in seconds, so ~15 frames a second decide the same as 30, twice as fast.
    Frames with a fresh face are always kept: the face model runs every other frame, and dropping
    exactly those would leave no brow readings at all."""
    def fresh(f: Frame) -> bool:
        return f.sig.face is not None and f.sig.face.fresh

    out: list[Frame] = []
    window: list[Frame] = []  # one kept frame per 1/fps window, one with a fresh face if there is one
    for f in frames:
        if window and (f.step != window[0].step or f.sig.t - window[0].sig.t >= 1.0 / fps - 1e-4):
            out.append(next((w for w in window if fresh(w)), window[0]))
            window = []
        window.append(f)
    if window:
        out.append(next((w for w in window if fresh(w)), window[0]))
    return out


def _brow_candidates(brows: Sequence[tuple[int, float, float]], up_mask: Sequence[bool],
                     down_mask: Sequence[bool]) -> list[dict[str, Any]]:
    """Per brow signal, (signal, rise) from the raised (75th percentile of the raise windows: a slow start
    doesn't drag it down) vs relaxed (median) readings; ordered best-separated signal first, then strict
    to loose share of the rise."""
    out = []
    for i, signal in ((1, "blend"), (2, "geo")):
        up_all = np.array([b[i] for b in brows if up_mask[b[0]]])
        up = up_all[up_all >= np.percentile(up_all, 50)] if len(up_all) >= 20 else up_all  # the raised half
        down = np.array([b[i] for b in brows if down_mask[b[0]]])
        if len(up) < 10 or len(down) < 10:
            continue
        rise = float(np.median(up) - np.median(down))
        if rise <= (0.05 if signal == "blend" else 0.01):
            continue
        sep = rise / (float(np.std(up) + np.std(down)) + 1e-6)
        out.append((sep, signal, rise))
    out.sort(reverse=True)
    return [{"signal": s, "rise": round(r, 4), "rise_share": share}
            for _, s, r in out for share in (0.75, 0.6, 0.45, 0.35)]


FOREHEAD_CANDIDATES: tuple[dict[str, Any], ...] = (
    {},  # as shipped
    {"occl_hold_s": 2.0},  # the "hand lost on the forehead" fallback, shorter
    {"occl_hold_s": 0.0},  # ... or off: the hand must stay visible
    {"occl_hold_s": 0.0, "min_tips": 3},
    {"occl_hold_s": 0.0, "min_tips": 3, "half_width": 0.4, "palm_expand": 0.05},  # a narrower forehead
    {"occl_hold_s": 0.0, "min_tips": 3, "half_width": 0.4, "palm_expand": 0.05, "max_speed": 0.5,
     "max_tip_spread": 1.0},
)
SIGN_CANDIDATES: tuple[dict[str, Any], ...] = (
    {},  # as shipped; only loosened if your sign isn't recognised
    {"min_score": 0.35},
    {"min_score": 0.35, "max_speed": 2.5},
)
L_CANDIDATES: tuple[dict[str, Any], ...] = ({}, {"max_speed": 2.5})


def analyse(frames: Sequence[Frame], overrides: Mapping[str, Mapping[str, Any]] | None = None,
            positive: Mapping[str, Sequence[str]] = POSITIVE, rest_steps: Sequence[str] = ("relax", "rest")) -> Result:
    """Pick per-user settings from recorded calibration frames (see the module docstring)."""
    overrides = overrides or {}
    frames = _thin(frames)
    base_truths, brows = _replay(frames, overrides, None)
    candidates: dict[str, list[dict[str, Any]]] = {
        FOREHEAD: [dict(c) for c in FOREHEAD_CANDIDATES],
        PEACE: [dict(c) for c in SIGN_CANDIDATES],
        L_SHAPE: [dict(c) for c in L_CANDIDATES],
    }
    raise_windows = _windows(frames, positive.get(EYEBROWS_UP, ()))
    up_mask = [any(w[i] for w in raise_windows) for i in range(len(frames))]
    late = _late(frames, positive.get(EYEBROWS_UP, ()))
    down_mask = [f.step in rest_steps and f.t >= (LATE_S if late[i] else REACT_S) and not up_mask[i]
                 for i, f in enumerate(frames)]
    brow_cands = _brow_candidates(brows, up_mask, down_mask)
    n = max([len(c) for c in candidates.values()] + [len(brow_cands)])
    # pass k applies candidate k of every gesture at once (they don't interact), so n replays in all
    passes: list[tuple[list[dict[str, bool]], dict[str, Any]]] = []
    for k in range(n):
        params = {g: c[k] for g, c in candidates.items() if k < len(c)}
        brow = brow_cands[k] if k < len(brow_cands) else None
        truths, _ = _replay(frames, overrides, {"params": params, "brows": brow})
        passes.append((truths, {"params": params, "brows": brow}))

    verdicts: dict[str, Verdict] = {}
    chosen_params: dict[str, dict[str, Any]] = {}
    for g in (FOREHEAD, PEACE, L_SHAPE):
        if g not in positive:
            continue
        options = [(k, *_evaluate(frames, t, g, positive)) for k, (t, cfg) in enumerate(passes) if g in cfg["params"]]
        k, fired, false_s = _pick(options, g)
        chosen_params[g] = passes[k][1]["params"][g]
        verdicts[g] = Verdict(g, fired, false_s, chosen_params[g], "tuned to you" if chosen_params[g] else "")
    brow_choice = None
    if brow_cands:
        options = [(k, *_evaluate(frames, passes[k][0], EYEBROWS_UP, positive)) for k in range(len(brow_cands))]
        k, fired, false_s = _pick(options, EYEBROWS_UP)
        note = "glasses-proof brow height" if brow_cands[k]["signal"] == "geo" else ""
        v = Verdict(EYEBROWS_UP, fired, false_s, dict(brow_cands[k]), note)
        brow_choice = brow_cands[k] if v.ok else None  # only a setting that works is worth keeping
        if not v.ok:
            v.params, v.note = {}, "kept your previous setting"
        verdicts[EYEBROWS_UP] = v
    else:
        fired, false_s = _evaluate(frames, base_truths, EYEBROWS_UP, positive)
        verdicts[EYEBROWS_UP] = Verdict(EYEBROWS_UP, False, false_s, note="couldn't see your brows move")
    for g in (TWO_THUMBS, HANDS_UP, COLLAR, COVER_FACE):
        if g in positive:
            fired, false_s = _evaluate(frames, base_truths, g, positive)
            verdicts[g] = Verdict(g, fired, false_s, margin=1.0)
    params = {g: p for g, p in chosen_params.items() if p}
    for g, v in verdicts.items():
        log.info("calibration %s: fired %s, elsewhere %.2f s, settings %s", g, v.fired, v.false_s, v.params)
    return Result(verdicts, brow_choice, params)


def _pick(options: Sequence[tuple[int, bool, float]], gesture: str) -> tuple[int, bool, float]:
    """The first candidate that fires and stays quiet elsewhere; else the quietest one that fires;
    else the quietest."""
    limit = FALSE_RUN_SHARE * FIRE_RUN_S.get(gesture, 0.6)
    for k, fired, false_s in options:
        if fired and false_s < limit:
            return k, fired, false_s
    firing = [o for o in options if o[1]]
    pool = firing or list(options)
    return min(pool, key=lambda o: (o[2], o[0]))


def frames_from_records(records: Iterable[Mapping[str, Any]], width: int = 1280, height: int = 720) -> list[Frame]:
    """``zap record`` / fixture frame records (with a "step" field) as calibration frames."""
    from zap.recording import signals_from_record

    out: list[Frame] = []
    step_t0: dict[str, float] = {}
    for i, rec in enumerate(records):
        sig = signals_from_record(rec, i, width, height)
        step = rec.get("step", "")
        t0 = step_t0.setdefault(step, sig.t)
        out.append(Frame(step, sig.t - t0, replace(sig, mask=None)))
    return out
