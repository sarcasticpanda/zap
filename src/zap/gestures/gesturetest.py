"""``zap test``: try every gesture on the real camera through the real detectors, and get a report.

Unlike the calibration, which judges its own steps, this asks the same PoseDetector + Arbiter that
run in a meeting: a gesture passes only if the app would really have fired it, and the time it took
is reported. It starts with a setup check (face seen, shoulders in the picture, a hand detectable),
then 15 s of normal talking where nothing may fire (false triggers), then each gesture with a time
limit. A failure says why: no face, no hand seen, the shape didn't match, a rule blocked it.

The landmarks of the whole test (no video) are saved next to the report so a failing gesture can
be replayed and fixed offline (``zap replay``).
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zap.gestures.poses import PoseDetector
from zap.gestures.types import (
    CANCEL,
    COLLAR,
    COVER_FACE,
    EYEBROWS_UP,
    FOREHEAD,
    HANDS_UP,
    L_SHAPE,
    PEACE,
    SNAP1,
    SNAP2,
    TWO_THUMBS,
    Intent,
)
from zap.recording import frame_record
from zap.signals import Signals
from zap.vision.visibility import Visibility

log = logging.getLogger(__name__)

LEAD_S = 3.0
RESULT_S = 1.6  # each result shows this long before the next step
SETUP_NEED_S = 1.5  # face + framing + a hand, seen this long (adds up)
SKIP_HINT_S = 10.0


@dataclass(frozen=True)
class GestureStep:
    name: str
    prompt: str
    expect: str | None  # the gesture that must fire (None: nothing may fire)
    seconds: float  # time limit (or, with expect None, how long to watch)
    kind: str = "gesture"  # "setup" | "quiet" | "gesture"


STEPS: tuple[GestureStep, ...] = (
    GestureStep("setup", "Setup check: sit as in a meeting, then raise one open hand beside your face", None, 30.0,
             "setup"),
    GestureStep("quiet", "Talk normally and look around for 15 s - no gestures", None, 15.0, "quiet"),
    GestureStep("peace", "PEACE sign beside your face, hold it", PEACE, 12.0),
    GestureStep("l_shape", "L sign beside your face (index up, thumb out), hold it", L_SHAPE, 12.0),
    GestureStep("thumbs", "THUMBS UP, hold it", TWO_THUMBS, 12.0),
    GestureStep("hands_up", "BOTH open hands up beside your head, hold", HANDS_UP, 12.0),
    GestureStep("collar", "Fingers at your COLLAR, hold", COLLAR, 12.0),
    GestureStep("forehead", "Hand flat on your FOREHEAD, hold", FOREHEAD, 12.0),
    GestureStep("eyebrows", "Hands down - RAISE your eyebrows and hold", EYEBROWS_UP, 12.0),
    GestureStep("cover", "COVER your face with both hands, then drop them", COVER_FACE, 12.0),
    GestureStep("palm", "Show an OPEN PALM to the camera, hold (ends a card)", CANCEL, 12.0),
    GestureStep("snap1", "SNAP your fingers once", SNAP1, 15.0),
    GestureStep("snap2", "SNAP twice, quickly", SNAP2, 15.0),
)


@dataclass
class StepResult:
    name: str
    expect: str | None
    passed: bool
    seconds: float  # time to fire (or how long the step ran)
    detail: str
    also_fired: list[str] = field(default_factory=list)  # other gestures that fired during the step
    face_pct: float = 0.0
    hand_pct: float = 0.0
    raw_pct: float = 0.0  # frames the detector saw the expected gesture


@dataclass
class TestView:
    title: str
    sub: str
    progress: float
    state: str  # "lead" | "wait" | "seen" | "tick" | "fail"


def _fired(intents: Sequence[Intent]) -> list[str]:
    out = []
    for it in intents:
        if it.kind == "trigger":
            out.append(it.gesture)
        elif it.kind == "cancel" and it.gesture == CANCEL:
            out.append(CANCEL)
    return out


class GestureTest:
    """Drive with :meth:`feed` every frame (after the arbiter ran) until it returns None."""

    def __init__(self, now: float, steps: Sequence[GestureStep] = STEPS) -> None:
        self.steps = list(steps)
        self.i = 0
        self.results: list[StepResult] = []
        self.records: list[dict[str, Any]] = []
        self._t_start = now
        self._begin(now + LEAD_S)
        self._lead_until = now + LEAD_S

    @property
    def done(self) -> bool:
        return self.i >= len(self.steps)

    @property
    def step(self) -> GestureStep | None:
        return None if self.done else self.steps[self.i]

    @property
    def card_needed(self) -> bool:
        """The open-palm step needs a card showing (the palm only ends cards)."""
        s = self.step
        return s is not None and s.expect == CANCEL and self._result_until is None

    def _begin(self, now: float) -> None:
        self._t0 = now
        self._last_t: float | None = None
        self._ok_s = 0.0
        self._result_until: float | None = None
        self._n = self._face = self._hand = self._raw = 0
        self._why: Counter[str] = Counter()
        self._also: list[str] = []

    def skip(self, now: float) -> None:
        s = self.step
        if s is not None and self._result_until is None:
            self._finish(now, False, now - self._t0, "skipped")

    def _finish(self, now: float, passed: bool, seconds: float, detail: str) -> None:
        s = self.steps[self.i]
        n = max(1, self._n)
        r = StepResult(s.name, s.expect, passed, round(seconds, 2), detail, sorted(set(self._also)),
                       round(100 * self._face / n), round(100 * self._hand / n), round(100 * self._raw / n))
        self.results.append(r)
        log.info("gesture test %s: %s (%s) %.1f s, face %d%%, hands %d%%, seen %d%%%s", s.name,
                 "PASS" if passed else "FAIL", detail, seconds, r.face_pct, r.hand_pct, r.raw_pct,
                 f", also fired {r.also_fired}" if r.also_fired else "")
        self._result_until = now + RESULT_S
        self._shown = r

    def feed(self, sig: Signals, det: PoseDetector, truths: Mapping[str, bool], intents: Sequence[Intent],
             vis: Visibility, now: float) -> TestView | None:
        if self.done:
            return None
        s = self.steps[self.i]
        n, total = self.i + 1, len(self.steps)
        rec = frame_record(sig, now - self._t_start, s.name)
        rec["truths"] = sorted(g for g, on in truths.items() if on)
        rec["intents"] = [f"{i.kind}:{i.gesture}" for i in intents]
        self.records.append(rec)
        if now < self._lead_until:
            return TestView("Gesture test: every gesture, through the real detection",
                            f"follow each prompt; N skips one, Esc stops  ...{int(self._lead_until - now) + 1}",
                            0.0, "lead")
        if self._result_until is not None:  # showing the last result
            r = self._shown
            if now >= self._result_until:
                self.i += 1
                self._begin(now)
                return None if self.done else self.feed(sig, det, truths, (), vis, now)
            if r.passed:
                return TestView(f"PASS: {s.prompt.split(',')[0]}", r.detail, 1.0, "tick")
            return TestView(f"FAIL: {s.prompt.split(',')[0]}", r.detail, 1.0, "fail")

        # one frame of this step
        dt = 0.0 if self._last_t is None else min(0.1, max(0.0, now - self._last_t))
        self._last_t = now
        self._n += 1
        self._face += sig.face is not None and sig.face.fresh
        self._hand += bool(sig.hands)
        if s.expect is not None and truths.get(s.expect):
            self._raw += 1
        fired = _fired(intents)
        elapsed = now - self._t0
        head = f"step {n} of {total}: "

        if s.kind == "setup":
            problems = vis.problems(now)
            hard = [t for sev, t in problems if sev == "error"]
            ok = vis.face_ok(now) and not hard and bool(sig.hands)
            if ok:
                self._ok_s += dt
            if self._ok_s >= SETUP_NEED_S:
                self._finish(now, True, elapsed, f"face, framing and a hand OK ({vis.summary(now)})")
                return self.feed(sig, det, truths, (), vis, now)
            if elapsed >= s.seconds:
                self._finish(now, False, elapsed, hard[0] if hard else "no hand seen: hold an open hand beside "
                             "your face, fully in the picture")
                return self.feed(sig, det, truths, (), vis, now)
            why = hard[0] if hard else (problems[0][1] if problems else (
                "now raise one open hand beside your face" if not sig.hands else "good, hold still ..."))
            return TestView(s.prompt, head + why, self._ok_s / SETUP_NEED_S, "seen" if ok else "wait")

        if s.kind == "quiet":
            self._also += fired
            if elapsed >= s.seconds:
                passed = not self._also
                detail = "nothing fired while you talked" if passed else \
                    f"FALSE TRIGGER while talking: {', '.join(sorted(set(self._also)))}"
                self._finish(now, passed, elapsed, detail)
                return self.feed(sig, det, truths, (), vis, now)
            sub = head + (f"oops, fired: {', '.join(self._also)}" if self._also else
                          f"keep talking ...{int(s.seconds - elapsed) + 1}")
            return TestView(s.prompt, sub, elapsed / s.seconds, "fail" if self._also else "seen")

        # a gesture step
        hit = s.expect in fired
        self._also += [g for g in fired if g != s.expect]
        if hit:
            self._finish(now, True, elapsed, f"fired after {elapsed:.1f} s")
            return self.feed(sig, det, truths, (), vis, now)
        reason = self._reason(s, sig, det, truths, vis, now)
        if reason:
            self._why[reason] += 1
        if elapsed >= s.seconds:
            top = self._why.most_common(1)[0][0] if self._why else "the gesture wasn't recognised"
            self._finish(now, False, elapsed, f"not fired in {s.seconds:.0f} s: {top}")
            return self.feed(sig, det, truths, (), vis, now)
        sub = head + (reason or "seen - hold it ...")
        if elapsed >= SKIP_HINT_S:
            sub += "  (N = skip)"
        seen = bool(s.expect and truths.get(s.expect))
        return TestView(s.prompt, sub, elapsed / s.seconds, "seen" if seen else "wait")

    def _reason(self, s: GestureStep, sig: Signals, det: PoseDetector, truths: Mapping[str, bool], vis: Visibility,
                now: float) -> str:
        """Why the expected gesture isn't firing right now, in words ('' = it's seen)."""
        g = s.expect
        if g is not None and truths.get(g):
            return ""
        needs_face = g in (EYEBROWS_UP, FOREHEAD, COLLAR, HANDS_UP, COVER_FACE)
        if needs_face and not vis.face_ok(now) and g != COVER_FACE:
            return "no face seen"
        if g != EYEBROWS_UP and not sig.hands:
            return "no hand seen (hand out of the picture, too dark, or moving fast)"
        if g == HANDS_UP and len(sig.hands) < 2:
            return f"only {len(sig.hands)} of 2 hands seen"
        if g == EYEBROWS_UP:
            b = det.debug.get(EYEBROWS_UP) or {}
            if sig.hands:
                return "a hand is in view (Jim needs hands down)"
            if b.get("speaking"):
                return "you seem to be talking (Jim waits for a still mouth)"
            if "level" in b:
                return f"brows {b['level']:.2f}, need {b['threshold']:.2f}"
            return "no brow reading"
        why = det.debug.get("sign_why", {}).get(g, "")
        if why:
            return why
        return "hand seen, but not this gesture's shape/position"


def save(test: GestureTest, folder: Path) -> Path:
    """Write report.json and signals.jsonl (landmarks only) into ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    report = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "passed": sum(r.passed for r in test.results),
        "total": len(test.results),
        "results": [r.__dict__ for r in test.results],
    }
    (folder / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    with open(folder / "signals.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in test.records)
    return folder


def format_report(test: GestureTest) -> str:
    lines = [f"Gesture test: {sum(r.passed for r in test.results)} of {len(test.results)} passed"]
    for r in test.results:
        mark = "PASS" if r.passed else "FAIL"
        extra = f"  (also fired: {', '.join(r.also_fired)})" if r.also_fired else ""
        lines.append(f"  {mark}  {r.name:<9} {r.detail}  [face {r.face_pct}%, hand {r.hand_pct}%, "
                     f"seen {r.raw_pct}%]{extra}")
    return "\n".join(lines)
