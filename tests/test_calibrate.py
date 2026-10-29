"""Calibration: each step waits for the action to be seen; the analysis tunes to the person; saving."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from zap.gestures import calibrate as cal
from zap.gestures.calibrate import Calibration, analyse
from zap.gestures.poses import PoseDetector
from zap.gestures.types import COLLAR, COVER_FACE, EYEBROWS_UP, FOREHEAD
from zap.signals import Face, Signals

W, H = 1280, 720
DT = 1 / 30
FW, CX, CY = 200.0, 640.0, 300.0
KEY_UV = {10: (0.0, -0.5), 168: (0.0, 0.0), 1: (0.0, 0.33), 152: (0.0, 0.78), 234: (-0.5, 0.05), 454: (0.5, 0.05),
          105: (-0.25, -0.12), 334: (0.25, -0.12), 145: (-0.2, 0.08), 374: (0.2, 0.08), 159: (-0.2, 0.0),
          386: (0.2, 0.0), 33: (-0.32, 0.02), 263: (0.32, 0.02)}


def face(level: float, jaw: float = 0.05, pitch: float = 0.0) -> Face:
    n = 478
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.sqrt(np.linspace(0.05, 1.0, n))
    u, v = 0.48 * r * np.cos(ang), 0.14 + 0.62 * r * np.sin(ang)
    for i, (uu, vv) in KEY_UV.items():
        u[i], v[i] = uu, vv
    lm = np.zeros((n, 3), np.float32)
    lm[:, 0], lm[:, 1] = CX + u * FW, CY + v * FW
    a = math.radians(pitch)
    m = np.eye(4)
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = math.cos(a), -math.sin(a), math.sin(a), math.cos(a)
    blend = {"browInnerUp": level, "browOuterUpLeft": level, "browOuterUpRight": level, "jawOpen": jaw}
    return Face(lm, blend, m, 0.0, True)


class Person:
    """Does each step's action (or nothing), frame by frame, through a real PoseDetector."""

    def __init__(self, raise_to: float = 0.85, rest: float = 0.25, talk_brows: float = 0.4) -> None:
        self.raise_to, self.rest, self.talk_brows = raise_to, rest, talk_brows
        self.rng = np.random.default_rng(3)

    def face_for(self, step: str, k: int) -> Face:
        noise = float(self.rng.normal(0, 0.015))
        if step in ("raise1", "raise2"):
            return face(self.raise_to + noise)
        if step == "talk":
            return face(self.talk_brows * abs(math.sin(0.2 * k)) + self.rest + noise, jaw=0.05 + 0.3 * abs(math.sin(0.4 * k)))
        if step == "look_down":
            return face(self.rest + 0.3 + noise, pitch=18.0)  # tilting the head lifts the brow readings
        if step == "look_up":
            return face(self.rest + 0.3 + noise, pitch=-14.0)
        return face(self.rest + noise)


def drive(run: Calibration, person: Person | None, t0: float = 0.0, seconds: float = 200.0, steps=None):
    """Feed frames until the run is done (or ``seconds`` pass). ``person`` None = sits there doing nothing."""
    det = PoseDetector()
    views = []
    k = 0
    t = t0
    while t < t0 + seconds:
        step = run.step.name if run.step is not None else "relax"
        if steps is not None and step not in steps:
            break
        f = person.face_for(step, k) if person is not None else face(0.25)
        sig = Signals(t, k, [], f, width=W, height=H)
        truths = det.update(sig, t)
        v = run.feed(sig, det, truths, t)
        if v is None:
            break
        views.append(v)
        k += 1
        t = t0 + k * DT
    return views


FACE_STEPS = ("relax", "talk", "look_down", "look_up", "raise1", "rest", "raise2")
FACE_ONLY = tuple(s for s in cal.STEPS if s.name in FACE_STEPS)


def test_doing_nothing_does_not_move_on():
    run = Calibration(0.0, FACE_ONLY, lead_s=0.5)
    views = drive(run, None, seconds=30.0)
    # relaxing passes, but sitting still never counts as talking: stuck on step 2, saying why
    assert run.step.name == "talk" and not run.done
    assert views[-1].state == "wait" and "mouth" in views[-1].sub
    assert "N = skip" in views[-1].sub  # after a while you may skip


def test_each_step_waits_for_its_action_then_moves_on():
    run = Calibration(0.0, FACE_ONLY, lead_s=0.5)
    views = drive(run, Person())
    assert run.done and not run.skipped
    states = [v.state for v in views]
    assert states.count("tick") > 5 and "seen" in states
    assert run.base["blend"] == pytest.approx(0.25, abs=0.03)  # your resting brows
    assert run.base["pitch"] == pytest.approx(0.0, abs=1.0)
    used = {f.step for f in run.frames if f.use}
    assert used == set(FACE_STEPS)


def test_looking_down_is_checked():
    run = Calibration(0.0, FACE_ONLY, lead_s=0.5)
    lazy = Person()
    lazy.face_for = lambda step, k: face(0.25, jaw=0.05 + 0.3 * abs(math.sin(0.4 * k)))  # never tilts the head
    drive(run, lazy, seconds=25.0)
    assert run.step.name == "look_down"


def test_a_raise_must_beat_your_own_resting_level():
    run = Calibration(0.0, FACE_ONLY, lead_s=0.5)
    timid = Person(raise_to=0.29)  # barely moves: not a raise
    drive(run, timid, seconds=40.0)
    assert run.step.name == "raise1" and "higher" in run.feed(
        Signals(99.0, 0, [], face(0.29), width=W, height=H), PoseDetector(), {}, 99.0).sub


def test_skip_moves_on_and_is_remembered():
    run = Calibration(0.0, FACE_ONLY, lead_s=0.5)
    drive(run, None, seconds=10.0)
    assert run.step.name == "talk"
    run.skip(10.0)
    assert run.step.name == "look_down" and "talk" in run.skipped


def test_retry_queues_the_failed_gestures_steps_once():
    run = Calibration(0.0, FACE_ONLY, lead_s=0.5)
    drive(run, Person())
    assert run.done
    assert run.retry([EYEBROWS_UP], 100.0)
    assert [s.name for s in run.steps[len(FACE_ONLY):]] == ["raise1", "rest", "raise2"]
    assert run.steps[-1].prompt.startswith("Once more")
    assert not run.retry([EYEBROWS_UP], 200.0)  # only once


def test_brows_tuned_to_fire_on_raises_and_not_while_talking_or_looking_down():
    run = Calibration(0.0, FACE_ONLY, lead_s=0.5)
    drive(run, Person())
    res = analyse(run.frames, positive={EYEBROWS_UP: ("raise1", "raise2")})
    v = res.verdicts[EYEBROWS_UP]
    assert v.ok, v.line()
    assert res.brows["signal"] == "blend" and res.brows["rise"] == pytest.approx(0.6, abs=0.08)


def test_brows_flying_up_while_talking_still_dont_fire_jim():
    run = Calibration(0.0, FACE_ONLY, lead_s=0.5)
    drive(run, Person(talk_brows=0.65))  # brows fly up while talking, as high as a raise
    res = analyse(run.frames, positive={EYEBROWS_UP: ("raise1", "raise2")})
    v = res.verdicts[EYEBROWS_UP]
    assert v.ok and v.false_fires == 0, v.line()  # the moving mouth gives talking away


def test_apply_calibration_keeps_config_values():
    det = PoseDetector({"forehead": {"min_tips": 4}})
    det.apply_calibration({"params": {"forehead": {"min_tips": 3, "occl_hold_s": 0.0}, "nope": {"x": 1},
                                      "peace": {"min_score": 0.35}}})
    assert det.params[FOREHEAD].min_tips == 4  # config.toml wins
    assert det.params[FOREHEAD].occl_hold_s == 0.0 and det.params["peace"].min_score == 0.35
    det.apply_calibration(None)
    assert det.params[FOREHEAD].occl_hold_s > 0  # back to the defaults


def test_saved_and_legacy_calibration_load(tmp_path, monkeypatch):
    from zap import app

    monkeypatch.setenv("ZAP_HOME", str(tmp_path))
    assert app._load_calibration() is None
    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration" / "eyebrows.json").write_text(json.dumps({"signal": "geo", "rise": 0.02}))
    assert app._load_calibration() == {"brows": {"signal": "geo", "rise": 0.02}}
    run = Calibration(0.0, FACE_ONLY, lead_s=0.5)
    drive(run, Person())
    res = analyse(run.frames, positive={EYEBROWS_UP: ("raise1", "raise2")})
    app._save_calibration(res.calibration())
    loaded = app._load_calibration()
    assert loaded["version"] == cal.VERSION and loaded["brows"] == res.brows and "report" in loaded


def test_every_step_has_a_check_and_every_gesture_a_step():
    names = {s.name for s in cal.STEPS}
    for s in cal.STEPS:
        assert hasattr(Calibration, "_check_" + s.check), s.name
    for g, steps in cal.POSITIVE.items():
        assert set(steps) <= names, g


# ---------------------------------------------------------------- a real user's recording
SITBACK = Path(__file__).parent / "fixtures_local" / "live_check_sitback.json"


def test_real_recording_forehead_collar_and_cover_check_out():
    """The user reacted ~2.5 s late to each instruction in this (timed) recording; shifted by that,
    Michael Scott, the collar and covering the face are recognised in their own steps and nowhere else."""
    if not SITBACK.is_file():
        pytest.skip("local recording not present")
    from zap.recording import load_frames

    recs = load_frames(SITBACK)
    bounds: list[tuple[float, str]] = []
    for r in recs:
        if not bounds or bounds[-1][1] != r.get("step", ""):
            bounds.append((float(r["t"]), r.get("step", "")))
    shifted = []
    for r in recs:
        t = float(r["t"]) - 2.5
        shifted.append({**r, "step": next((s for t0, s in reversed(bounds) if t >= t0), bounds[0][1])})
    pos = {EYEBROWS_UP: ("raise1", "raise2", "raise3"), FOREHEAD: ("forehead",), COLLAR: ("collar",),
           COVER_FACE: ("cover", "uncover")}
    res = analyse(cal.frames_from_records(shifted), positive=pos, rest_steps=("relax", "rest1", "rest2", "rest3"))
    for g in (FOREHEAD, COLLAR, COVER_FACE):
        assert res.verdicts[g].ok, res.verdicts[g].line()
    assert res.params == {}  # the shipped forehead settings already work for this user
