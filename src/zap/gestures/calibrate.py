"""Ctrl+Alt+C / first use: calibrate every gesture to the person in front of the camera.

An interactive check, step by step. Each step waits until the camera really
sees you do it (your head tilted down, your brows up from YOUR resting level,
your hand on your forehead, each sign...), fills a bar while you hold it, ticks
it off and only then moves on; if it isn't seen, it tells you why and keeps
waiting (N skips). The preview draws the tracked points on your face and hands
meanwhile, so you can see what ZAP sees.

Only the frames where the action was confirmed are used. They (landmarks only,
kept in memory, never written to disk) are replayed through the real
:class:`PoseDetector` with candidate settings; per gesture the first candidate,
in order of preference, that fires in its own step and never in the others
(talking, looking down and up, touching your face, the other gestures) is kept.
Gestures that still don't work get one more try, then the result is saved.

The analysis is a pure function of the recorded frames (:func:`analyse`), so it
also runs on ``zap record`` recordings and in tests.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from zap.gestures.poses import (
    PoseDetector,
    _levels,
    _nose_ratio,
    l_shape,
    peace_shape,
    peace_shape_strict,
    thumb_up_shape,
    usable,
)
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
from zap.signals import Face, Hand, Signals
from zap.vision.geometry import face_frame, head_yaw_deg, is_open

log = logging.getLogger(__name__)

VERSION = 3
WAIT = "wait"  # frames recorded while waiting for an action: not analysed
REACT_S = 1.0  # recordings with timed steps: skip this long of each step (reacting to the prompt)
LATE_S = 2.0  # ... and of the step after a gesture's own step (people let go of a pose late)
LATE_POS_S = 1.0  # recordings: a gesture shown this late (into the next step) still counts for its own step
ANALYSE_FPS = 12.0
YIELD_EVERY = 25  # frames between short sleeps while replaying
LEAD_S = 3.0  # "get ready" before the first step
TICK_S = 0.8  # the green tick shows this long before the next step
HINT_S = 6.0  # waiting this long: show the hint
SKIP_S = 12.0  # ... and this long: offer to skip
MAX_DT = 0.1  # a frame never counts for more than this (a stalled camera doesn't fill the bar)


# ---------------------------------------------------------------- the steps
@dataclass(frozen=True)
class Step:
    name: str
    prompt: str
    need_s: float  # seconds of the action actually seen (they add up; gaps are fine)
    hint: str
    check: str  # Calibration method: "_check_<check>"


STEPS: tuple[Step, ...] = (
    Step("relax", "Look at the camera, face relaxed", 3.0,
         "sit so your whole face is in the picture, facing the camera", "relax"),
    Step("talk", "Talk: say anything out loud (count to ten)", 4.0,
         "move your mouth while you talk", "talk"),
    Step("look_down", "Look DOWN at your keyboard", 1.5, "tilt your head down, like reading", "look_down"),
    Step("look_up", "Now look UP at the ceiling", 1.5, "tilt your head back, up", "look_up"),
    Step("raise1", "RAISE your eyebrows as high as you can, hold", 1.5,
         "higher - like you're very surprised, looking at the camera", "raise"),
    Step("rest", "Relax your eyebrows", 1.0, "let them drop back down", "rest"),
    Step("raise2", "RAISE your eyebrows again, hold", 1.5,
         "higher - like you're very surprised, looking at the camera", "raise"),
    Step("forehead", "Put your hand flat on your FOREHEAD and hold (Michael Scott)", 2.0,
         "palm on your forehead, face still in view", "forehead"),
    Step("touch", "Touch your face naturally: rub your eyes, fix your glasses or hair", 4.0,
         "touch your face with your hands, like you normally would", "touch"),
    Step("peace", "PEACE sign beside your face, hold", 2.0,
         "two fingers up in a V, the others folded, hand beside your face", "peace"),
    Step("l_shape", "L sign beside your face: index finger up, thumb out, hold", 2.0,
         "index straight up, thumb straight out sideways, other fingers folded", "l_shape"),
    Step("thumbs", "THUMBS UP, hold", 2.0, "fist closed, thumb straight up", "thumbs"),
    Step("hands_up", "BOTH open hands up beside your head, hold", 1.5,
         "both palms open, beside your head (not above it)", "hands_up"),
    Step("collar", "Fingers at your COLLAR, hold", 1.5, "pinch your collar just under your chin", "collar"),
    Step("cover", "COVER your face with both hands", 1.0, "both hands over your whole face", "cover"),
    Step("uncover", "...now drop your hands", 0.4, "take both hands away from your face", "uncover"),
)
STEP_BY_NAME = {s.name: s for s in STEPS}

# which steps each gesture should fire in (everywhere else it must not)
POSITIVE: dict[str, tuple[str, ...]] = {
    EYEBROWS_UP: ("raise1", "raise2"),
    FOREHEAD: ("forehead",),
    PEACE: ("peace",),
    L_SHAPE: ("l_shape",),
    TWO_THUMBS: ("thumbs",),
    HANDS_UP: ("hands_up",),
    COLLAR: ("collar",),
    COVER_FACE: ("cover", "uncover"),
}
RETRY_STEPS = {EYEBROWS_UP: ("raise1", "rest", "raise2"), COVER_FACE: ("cover", "uncover")}  # else POSITIVE
REST_STEPS = ("relax", "rest")  # your resting brows
EVENTS = (COVER_FACE,)  # one-frame truths: fired = seen at least once in its steps
NAMES = {EYEBROWS_UP: "Jim (eyebrows)", FOREHEAD: "Michael Scott (forehead)", PEACE: "peace sign",
         L_SHAPE: "L sign", TWO_THUMBS: "thumbs up", HANDS_UP: "both hands up", COLLAR: "collar",
         COVER_FACE: "cover face"}
HINTS = {g: STEP_BY_NAME[s[0]].hint for g, s in POSITIVE.items()}
FIRE_RUN_S = {EYEBROWS_UP: 0.6, FOREHEAD: 0.7, PEACE: 0.7, L_SHAPE: 0.7, TWO_THUMBS: 0.7, HANDS_UP: 0.4,
              COLLAR: 0.6}  # a held pose must be seen this long without a gap to fire
MAX_GAP_S = 0.15  # gaps shorter than this don't break a run (the hand model drops frames)
FALSE_RUN_SHARE = 0.5  # elsewhere, runs must stay under this share of the firing time

LOOK_DEG = 12.0  # looking down / up: the head tilts this far from your resting pose
LOOK_BACK_DEG = 8.0  # ... the other way for the second one
RAISE_NEED = {"blend": (0.10, 0.25), "geo": (0.010, 0.025)}  # a raise: 3 x your resting jitter, within these
TALK_JAW = 0.06  # talking: the mouth opens and closes this much within TALK_WINDOW_S (a real user: p50 0.07)
TALK_WINDOW_S = 0.8


@dataclass
class Frame:
    step: str
    t: float  # seconds into the step
    sig: Signals
    use: bool = True  # analysed (False while waiting for the action, or reacting to a prompt)


@dataclass
class View:
    """What the preview shows for the calibration this frame."""

    title: str
    sub: str
    progress: float  # 0..1 of the current step
    n: int  # step number (1-based) and how many there are
    total: int
    state: str  # "lead" | "wait" | "seen" | "tick"


# ---------------------------------------------------------------- the interactive run
class Calibration:
    """The steps, one after another; ``feed`` every frame until it returns None, then :func:`analyse` the
    ``frames``. :meth:`retry` queues another try of some steps."""

    def __init__(self, now: float, steps: Sequence[Step] = STEPS, lead_s: float = LEAD_S) -> None:
        self.steps: list[Step] = list(steps)
        self.i = 0
        self.frames: list[Frame] = []
        self.skipped: set[str] = set()
        self.base: dict[str, float] = {}  # your resting pose, from the "relax" step
        self.retried = False
        self._samples: dict[str, list[float]] = {k: [] for k in ("blend", "geo", "pitch")}
        self._jaw: deque[tuple[float, float]] = deque()
        self._look_sign = 0.0
        self._start(now + lead_s)
        self._lead_until = now + lead_s

    # ------------------------------------------------------------ driving
    @property
    def done(self) -> bool:
        return self.i >= len(self.steps)

    @property
    def step(self) -> Step | None:
        return None if self.done else self.steps[self.i]

    def _start(self, now: float) -> None:
        self._t0 = now
        self._ok_s = 0.0
        self._last_t: float | None = None
        self._tick_until: float | None = None

    def skip(self, now: float) -> None:
        """N in the preview: give up on this step."""
        if self.step is not None and self._tick_until is None:
            log.info("calibration: step %s skipped", self.step.name)
            self.skipped.add(self.step.name)
            self._next(now)

    def retry(self, gestures: Iterable[str], now: float) -> bool:
        """Queue one more try of these gestures' steps; False if there is nothing to retry."""
        names: list[str] = []
        for g in gestures:
            for s in RETRY_STEPS.get(g, POSITIVE.get(g, ())):
                if s not in names and s not in self.skipped:
                    names.append(s)
        if not names or self.retried:
            return False
        self.retried = True
        self.steps += [replace(STEP_BY_NAME[s], prompt="Once more: " + STEP_BY_NAME[s].prompt) for s in names]
        self._start(now + 1.0)
        self._lead_until = now + 1.0
        return True

    def _next(self, now: float) -> None:
        self.i += 1
        self._start(now)

    def feed(self, sig: Signals, det: PoseDetector, truths: Mapping[str, bool], now: float) -> View | None:
        """Check this frame against the current step and record it. None once every step is done."""
        if self.done:
            return None
        step = self.steps[self.i]
        n, total = self.i + 1, len(self.steps)
        if now < self._lead_until:
            self.frames.append(Frame(WAIT, 0.0, _lean(sig), False))
            return View("Get ready: calibrating ZAP to YOU", f"follow each prompt; the bar fills when ZAP sees it "
                        f"(Esc cancels)  ...{math.ceil(self._lead_until - now)}", 0.0, n, total, "lead")
        if self._tick_until is not None:  # the step is done: show the tick, then the next step
            self.frames.append(Frame(WAIT, now - self._t0, _lean(sig), False))
            if now >= self._tick_until:
                self._next(now)
                return self.feed(sig, det, truths, now) if not self.done else None
            return View(step.prompt, "got it!", 1.0, n, total, "tick")
        ok, note = getattr(self, "_check_" + step.check)(sig, det, truths, now)
        dt = 0.0 if self._last_t is None else min(MAX_DT, max(0.0, now - self._last_t))
        self._last_t = now
        if ok:
            self._ok_s += dt
            if step.check == "relax":
                self._sample_rest(sig, det)
        self.frames.append(Frame(step.name if ok else WAIT, now - self._t0, _lean(sig), ok))
        progress = min(1.0, self._ok_s / step.need_s)
        if progress >= 1.0:
            if step.check == "relax":
                self._set_base()
            self._tick_until = now + TICK_S
            log.info("calibration: step %s done (%.1f s)", step.name, now - self._t0)
            return View(step.prompt, "got it!", 1.0, n, total, "tick")
        waited = now - self._t0
        if ok:
            sub = "yes - hold it ..." if progress < 0.99 else "got it!"
        elif waited >= HINT_S:
            sub = (note or step.hint) + ("   (N = skip this one)" if waited >= SKIP_S else "")
        else:
            sub = note or "waiting for you ..."
        return View(step.prompt, sub, progress, n, total, "seen" if ok else "wait")

    # ------------------------------------------------------------ your resting pose
    def _sample_rest(self, sig: Signals, det: PoseDetector) -> None:
        if sig.face is None or not sig.face.fresh:
            return
        level, geo = det.brow_values
        self._samples["blend"].append(level)
        self._samples["geo"].append(geo)
        self._samples["pitch"].append(_pitch(sig.face))

    def _set_base(self) -> None:
        for k, v in self._samples.items():
            if v:
                a = np.asarray(v, np.float64)
                self.base[k] = float(np.median(a))
                self.base[k + "_sd"] = float(np.std(a))
        log.info("calibration: resting pose %s", {k: round(v, 3) for k, v in self.base.items()})

    def _rise(self, det: PoseDetector) -> float:
        """How far your brows are up from rest, as a share of a clear raise (>= 1: raised)."""
        level, geo = det.brow_values
        best = 0.0
        for key, value in (("blend", level), ("geo", geo)):
            if key not in self.base:
                continue
            lo, hi = RAISE_NEED[key]
            need = min(hi, max(lo, 3.0 * self.base[key + "_sd"]))
            best = max(best, (value - self.base[key]) / need)
        return best

    def _tilt(self, face: Face) -> float:
        return _pitch(face) - self.base.get("pitch", _pitch(face))

    # ------------------------------------------------------------ checks: (seen now, what's wrong)
    def _check_relax(self, sig, det, truths, now) -> tuple[bool, str]:
        f = sig.face
        if f is None:
            return False, "I can't see your face: sit in front of the camera"
        if abs(head_yaw_deg(f)) > 15.0:
            return False, "look straight at the camera"
        if sig.hands:
            return False, "hands down, please"
        return f.fresh, ""

    def _check_talk(self, sig, det, truths, now) -> tuple[bool, str]:
        f = sig.face
        if f is None:
            return False, "I can't see your face"
        if f.fresh:
            self._jaw.append((now, float(f.blend.get("jawOpen", 0.0))))
        while self._jaw and now - self._jaw[0][0] > TALK_WINDOW_S:
            self._jaw.popleft()
        jaws = [j for _, j in self._jaw]
        talking = len(jaws) >= 3 and max(jaws) - min(jaws) >= TALK_JAW
        return talking, "" if talking else "talk out loud - I can't see your mouth move"

    def _check_look_down(self, sig, det, truths, now) -> tuple[bool, str]:
        f = sig.face
        if f is None:
            return False, "keep your face in the picture"
        tilt = self._tilt(f)
        if abs(tilt) >= LOOK_DEG:
            self._look_sign = self._look_sign or float(np.sign(tilt))
            return True, ""
        return False, f"tilt your head down more ({abs(tilt):.0f} of {LOOK_DEG:.0f} degrees)"

    def _check_look_up(self, sig, det, truths, now) -> tuple[bool, str]:
        f = sig.face
        if f is None:
            return False, "keep your face in the picture"
        tilt = self._tilt(f) * -(self._look_sign or 1.0)  # the other way from looking down
        if tilt >= LOOK_BACK_DEG:
            return True, ""
        return False, f"tilt your head back, up ({max(0.0, tilt):.0f} of {LOOK_BACK_DEG:.0f} degrees)"

    def _check_raise(self, sig, det, truths, now) -> tuple[bool, str]:
        f = sig.face
        if f is None or not f.fresh:
            return False, "keep your face in view"
        if sig.hands:
            return False, "hands down - just your eyebrows"
        if abs(self._tilt(f)) >= LOOK_BACK_DEG * 0.75:
            return False, "keep your head still, looking at the camera - just your eyebrows"
        r = self._rise(det)
        return r >= 1.0, "" if r >= 1.0 else f"higher! ({max(0, int(r * 100))}% of the way)"

    def _check_rest(self, sig, det, truths, now) -> tuple[bool, str]:
        f = sig.face
        if f is None or not f.fresh:
            return False, "keep your face in view"
        return self._rise(det) < 0.75, "let your eyebrows drop"  # brows settle slowly after a big raise

    def _check_forehead(self, sig, det, truths, now) -> tuple[bool, str]:
        face = det.face
        if face is None:
            return False, "keep your face in the picture"
        if det.occluded or truths.get(FOREHEAD):
            return True, ""
        ff = face_frame(face)
        lv = _levels(face, ff)
        for h in sig.hands:
            u, v = ff.to_uv(h.palm_center[None])[0]
            if abs(u) <= 0.8 and lv.top - 0.6 <= v <= lv.eye + 0.05:
                return True, ""
        return False, "I don't see a hand on your forehead" if not sig.hands else "higher: on your forehead"

    def _check_touch(self, sig, det, truths, now) -> tuple[bool, str]:
        face = det.face
        if face is None:
            return bool(sig.hands), ""  # a hand in front of the face hides it: that's touching
        x0, y0, x1, y1 = face.bbox()
        m = 0.5 * face.width
        near = any(x0 - m <= h.palm_center[0] <= x1 + m and y0 - m <= h.palm_center[1] <= y1 + m for h in sig.hands)
        return near or det.occluded, "" if near else "bring your hands to your face"

    def _sign(self, sig: Signals, det: PoseDetector, test: Callable[[Hand], bool], name: str) -> tuple[bool, str]:
        hands = [h for h in sig.hands if usable(h, sig.width, sig.height, det.params["general"])]
        if not sig.hands:
            return False, "I can't see your hand: hold it up beside your face, whole hand in the picture"
        if not hands:
            return False, "move your hand fully into the picture"
        if any(test(h) for h in hands):
            return True, ""
        why = det.debug.get("sign_why", {}).get(name, "")
        return False, f"I see your hand, but not the sign{f' ({why})' if why else ''}"

    def _check_peace(self, sig, det, truths, now) -> tuple[bool, str]:
        return self._sign(sig, det, lambda h: (h.gesture == "Victory" and h.gesture_score >= 0.3 and peace_shape(h))
                          or peace_shape_strict(h), PEACE)

    def _check_l_shape(self, sig, det, truths, now) -> tuple[bool, str]:
        return self._sign(sig, det, l_shape, L_SHAPE)

    def _check_thumbs(self, sig, det, truths, now) -> tuple[bool, str]:
        return self._sign(sig, det, lambda h: (h.gesture == "Thumb_Up" and h.gesture_score >= 0.3)
                          or thumb_up_shape(h), TWO_THUMBS)

    def _check_hands_up(self, sig, det, truths, now) -> tuple[bool, str]:
        if truths.get(HANDS_UP):
            return True, ""
        face = det.face
        if len(sig.hands) < 2:
            return False, f"I see {len(sig.hands)} of your 2 hands"
        if face is None:
            return False, "keep your face in the picture, between your hands"
        ff = face_frame(face)
        lv = _levels(face, ff)
        uv = ff.to_uv(np.stack([h.palm_center for h in sig.hands[:2]]))
        ok = (all(is_open(h) for h in sig.hands[:2]) and uv[:, 0].min() < -0.5 < 0.5 < uv[:, 0].max()
              and bool(np.all((uv[:, 1] >= lv.top - 1.2) & (uv[:, 1] <= lv.chin + 0.4))))
        return ok, "" if ok else "open palms, one each side of your head"

    def _check_collar(self, sig, det, truths, now) -> tuple[bool, str]:
        if truths.get(COLLAR):
            return True, ""
        face = det.face
        if not sig.hands:
            return False, "I can't see your hand: fingers at your collar, in the picture"
        if face is None:
            return False, "keep your face in the picture"
        ff = face_frame(face)
        lv = _levels(face, ff)
        for h in sig.hands:
            tips = ff.to_uv(h.lm[[8, 12, 16, 20], :2])
            if int(((np.abs(tips[:, 0]) <= 1.0) & (tips[:, 1] >= lv.chin) & (tips[:, 1] <= lv.chin + 0.9)).sum()) >= 2:
                return True, ""
        return False, "lower: at your collar, just under your chin"

    def _check_cover(self, sig, det, truths, now) -> tuple[bool, str]:
        ok = det.covering
        return ok, "" if ok else ("both hands over your face" if len(sig.hands) < 2 else "cover your whole face")

    def _check_uncover(self, sig, det, truths, now) -> tuple[bool, str]:
        ok = bool(truths.get(COVER_FACE)) or (sig.face is not None and sig.face.fresh and not det.covering)
        return ok, ""


def _lean(sig: Signals) -> Signals:
    return Signals(sig.t, sig.seq, list(sig.hands), sig.face, None, False, sig.width, sig.height)


def _pitch(face: Face) -> float:
    """Head tilt up/down in degrees (the face model's pose; the nose position if it has none)."""
    m = face.matrix
    if m is not None and getattr(m, "shape", None) == (4, 4):
        return float(math.degrees(math.atan2(m[2, 1], m[2, 2])))
    return 100.0 * _nose_ratio(face)  # roughly degrees for small tilts


# ---------------------------------------------------------------- analysis
@dataclass
class Verdict:
    gesture: str
    fired: bool  # fired in its own step(s)
    false_s: float  # longest stretch it was on anywhere else, seconds
    params: dict[str, Any] = field(default_factory=dict)  # the chosen tuning (detector side)
    note: str = ""
    margin: float = FALSE_RUN_SHARE  # tuned gestures keep a safety margin; checked ones just mustn't fire
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return self.fired and self.false_s < self.margin * FIRE_RUN_S.get(self.gesture, 0.6)

    def line(self) -> str:
        name = NAMES.get(self.gesture, self.gesture)
        if self.skipped:
            return f"{name}: skipped"
        if self.ok:
            return f"{name}: OK" + (f" ({self.note})" if self.note else "")
        if not self.fired:
            return f"{name}: NOT RECOGNISED - {HINTS.get(self.gesture, 'try again')}"
        if self.gesture == EYEBROWS_UP:
            return f"{name}: your brows go that high while you talk or look down too - raise them higher"
        return f"{name}: also fired when you weren't doing it ({self.note or 'kept the strictest setting'})"


@dataclass
class Result:
    verdicts: dict[str, Verdict]
    brows: dict[str, Any] | None
    params: dict[str, dict[str, Any]]

    @property
    def failed(self) -> list[str]:
        return [g for g, v in self.verdicts.items() if not v.ok and not v.skipped]

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


def _order(frames: Sequence[Frame]) -> list[str]:
    return list(dict.fromkeys(f.step for f in frames if f.step != WAIT))


def _windows(frames: Sequence[Frame], steps: Sequence[str]) -> list[list[bool]]:
    """Per positive step, the frames where the gesture is expected: the step's confirmed frames, and the
    first LATE_POS_S of the next step (in timed recordings people often get there a moment late)."""
    order = _order(frames)
    out = []
    for s in (s for s in order if s in steps):
        nxt = order[order.index(s) + 1] if order.index(s) + 1 < len(order) else None
        out.append([f.use and (f.step == s or (f.step == nxt and f.t < LATE_POS_S)) for f in frames])
    return out


def _elsewhere(frames: Sequence[Frame], steps: Sequence[str]) -> list[bool]:
    """Frames where the gesture must not fire: other steps' confirmed frames, forgiving the start of the
    step right after one of its own (letting go of a pose late)."""
    order = _order(frames)
    after = {order[i + 1] for i, s in enumerate(order[:-1]) if s in steps}
    pos = set(steps)
    return [f.use and f.step != WAIT and f.step not in pos and not (f.step in after and f.t < LATE_S)
            for f in frames]


def _evaluate(frames: Sequence[Frame], truths: Sequence[Mapping[str, bool]], gesture: str,
              positive: Mapping[str, Sequence[str]]) -> tuple[bool, float]:
    """(fired in its own steps, longest stretch it was on anywhere else). With several own steps (two
    eyebrow raises), at least half of them must fire."""
    on = [bool(tr.get(gesture, False)) for tr in truths]
    windows = _windows(frames, positive.get(gesture, ()))
    if not windows:
        return False, 0.0
    elsewhere = _elsewhere(frames, positive.get(gesture, ()))
    if gesture in EVENTS:  # it happens once (e.g. as the hands come off the face), in any of its steps
        fired = any(v and w for win in windows for v, w in zip(on, win, strict=True))
        return fired, (1.0 if any(v and e for v, e in zip(on, elsewhere, strict=True)) else 0.0)
    need = FIRE_RUN_S.get(gesture, 0.6)
    hits = sum(_runs(frames, on, win) + 1e-6 >= need for win in windows)
    return hits >= max(1, math.ceil(len(windows) / 2)), _runs(frames, on, elsewhere)


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
    Frames with a fresh face are preferred: the face model may run every other frame, and dropping
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
    """Per brow signal, (signal, rise) from the raised (the upper half of the raise frames: a slow start
    doesn't drag it down) vs relaxed (median) readings; ordered best-separated signal first, then strict
    to loose share of the rise."""
    out = []
    for i, signal in ((1, "blend"), (2, "geo")):
        up_all = np.array([b[i] for b in brows if up_mask[b[0]]])
        up = up_all[up_all >= np.percentile(up_all, 50)] if len(up_all) >= 20 else up_all
        down = np.array([b[i] for b in brows if down_mask[b[0]]])
        if len(up) < 5 or len(down) < 10:
            continue
        rise = float(np.median(up) - np.median(down))
        if rise <= (0.05 if signal == "blend" else 0.008):
            continue
        sep = rise / (float(np.std(up) + np.std(down)) + 1e-6)
        out.append((sep, signal, rise))
    out.sort(reverse=True)
    return [{"signal": s, "rise": round(r, 4), "rise_share": share} for _, s, r in out for share in (0.75, 0.55)]


BROW_PARAMS: tuple[dict[str, Any], ...] = ({}, {"max_tilt": 0.035})  # also stricter about head tilt
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
            positive: Mapping[str, Sequence[str]] = POSITIVE, rest_steps: Sequence[str] = REST_STEPS,
            skipped: Iterable[str] = ()) -> Result:
    """Pick per-user settings from recorded calibration frames (see the module docstring)."""
    overrides = overrides or {}
    skipped = set(skipped)
    frames = _thin(frames)
    base_truths, brows = _replay(frames, overrides, None)
    candidates: dict[str, list[dict[str, Any]]] = {
        FOREHEAD: [dict(c) for c in FOREHEAD_CANDIDATES],
        PEACE: [dict(c) for c in SIGN_CANDIDATES],
        L_SHAPE: [dict(c) for c in L_CANDIDATES],
    }
    raise_windows = _windows(frames, positive.get(EYEBROWS_UP, ()))
    up_mask = [any(w[i] for w in raise_windows) for i in range(len(frames))]
    rest = set(rest_steps)
    down_mask = [f.use and f.step in rest and not up_mask[i] for i, f in enumerate(frames)]
    brow_cands = [(b, p) for b in _brow_candidates(brows, up_mask, down_mask) for p in BROW_PARAMS]
    n = max([len(c) for c in candidates.values()] + [len(brow_cands)])
    # pass k applies candidate k of every gesture at once (they don't interact), so n replays in all
    passes: list[tuple[list[dict[str, bool]], dict[str, Any]]] = []
    for k in range(n):
        params = {g: c[k] for g, c in candidates.items() if k < len(c)}
        brow = None
        if k < len(brow_cands):
            brow, params[EYEBROWS_UP] = brow_cands[k]
        truths, _ = _replay(frames, overrides, {"params": params, "brows": brow})
        passes.append((truths, {"params": params, "brows": brow}))

    verdicts: dict[str, Verdict] = {}
    chosen: dict[str, dict[str, Any]] = {}
    for g in (FOREHEAD, PEACE, L_SHAPE):
        if g not in positive:
            continue
        options = [(k, *_evaluate(frames, t, g, positive)) for k, (t, cfg) in enumerate(passes) if g in cfg["params"]]
        k, fired, false_s = _pick(options, g)
        chosen[g] = passes[k][1]["params"][g]
        verdicts[g] = Verdict(g, fired, false_s, chosen[g], "tuned to you" if chosen[g] else "")
    brow_choice = None
    if EYEBROWS_UP in positive:
        if brow_cands:
            options = [(k, *_evaluate(frames, passes[k][0], EYEBROWS_UP, positive)) for k in range(len(brow_cands))]
            k, fired, false_s = _pick(options, EYEBROWS_UP)
            b, p = brow_cands[k]
            v = Verdict(EYEBROWS_UP, fired, false_s, {**b, **p}, "glasses-proof brow height" if b["signal"] == "geo"
                        else "")
            if v.ok:
                brow_choice = b
                if p:
                    chosen[EYEBROWS_UP] = p
            else:
                v.params, v.note = {}, "kept your previous setting"
            verdicts[EYEBROWS_UP] = v
        else:
            fired, false_s = _evaluate(frames, base_truths, EYEBROWS_UP, positive)
            verdicts[EYEBROWS_UP] = Verdict(EYEBROWS_UP, False, false_s, note="couldn't see your brows move")
    for g in (TWO_THUMBS, HANDS_UP, COLLAR, COVER_FACE):
        if g in positive:
            fired, false_s = _evaluate(frames, base_truths, g, positive)
            verdicts[g] = Verdict(g, fired, false_s, margin=1.0)
    for g, v in verdicts.items():
        v.skipped = bool(set(positive.get(g, ())) & skipped) and not v.ok
        log.info("calibration %s: fired %s, elsewhere %.2f s, settings %s%s", g, v.fired, v.false_s, v.params,
                 " (skipped)" if v.skipped else "")
    params = {g: p for g, p in chosen.items() if p}
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
    """Frame records with a "step" field (``zap record``, fixtures, saved calibrations) as calibration frames.
    Steps were timed in recordings, so the first REACT_S of each is skipped; "wait" frames never count."""
    from zap.recording import signals_from_record

    out: list[Frame] = []
    step_t0: dict[str, float] = {}
    prev = None
    for i, rec in enumerate(records):
        sig = signals_from_record(rec, i, width, height)
        step = rec.get("step", "")
        if step != prev:
            step_t0[step] = sig.t
            prev = step
        t = sig.t - step_t0[step]
        use = rec.get("use", t >= REACT_S) and step != WAIT
        out.append(Frame(step, t, replace(sig, mask=None), bool(use)))
    return out
