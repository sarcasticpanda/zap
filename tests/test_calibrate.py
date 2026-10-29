"""Ctrl+Alt+C calibration: the guided steps, the analysis (synthetic and a real user's recording), saving."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from zap.gestures import calibrate as cal
from zap.gestures.calibrate import CalibrationRun, Frame, Step, analyse
from zap.gestures.poses import PoseDetector
from zap.gestures.types import COLLAR, COVER_FACE, EYEBROWS_UP, FOREHEAD
from zap.signals import Face, Signals

W, H = 1280, 720
DT = 1 / 30
FW, CX, CY = 200.0, 640.0, 300.0
KEY_UV = {10: (0.0, -0.5), 168: (0.0, 0.0), 1: (0.0, 0.33), 152: (0.0, 0.78), 234: (-0.5, 0.05), 454: (0.5, 0.05),
          105: (-0.25, -0.12), 334: (0.25, -0.12), 145: (-0.2, 0.08), 374: (0.2, 0.08), 159: (-0.2, 0.0),
          386: (0.2, 0.0), 33: (-0.32, 0.02), 263: (0.32, 0.02)}


def face(level: float, jaw: float = 0.05) -> Face:
    n = 478
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.sqrt(np.linspace(0.05, 1.0, n))
    u, v = 0.48 * r * np.cos(ang), 0.14 + 0.62 * r * np.sin(ang)
    for i, (uu, vv) in KEY_UV.items():
        u[i], v[i] = uu, vv
    lm = np.zeros((n, 3), np.float32)
    lm[:, 0], lm[:, 1] = CX + u * FW, CY + v * FW
    blend = {"browInnerUp": level, "browOuterUpLeft": level, "browOuterUpRight": level, "jawOpen": jaw}
    return Face(lm, blend, None, 0.0, True)


def session(levels: dict[str, object], seconds: dict[str, float] | None = None) -> list[Frame]:
    """Frames for the steps in ``levels`` (a brow level, or a function of seconds into the step)."""
    rng = np.random.default_rng(3)
    frames, t = [], 0.0
    for step, level in levels.items():
        n = round((seconds or {}).get(step, 3.5) / DT)
        for k in range(n):
            lv = level(k * DT) if callable(level) else level
            f = face(float(np.clip(lv + rng.normal(0, 0.015), 0, 1)))
            frames.append(Frame(step, k * DT, Signals(t, len(frames), [], f, width=W, height=H)))
            t += DT
    return frames


RAISE = lambda s: 0.25 if s < 0.6 else 0.85
TALK = lambda s: 0.3 + 0.18 * abs(np.sin(2.1 * s))
POS = {EYEBROWS_UP: ("raise1", "raise2")}
ORDER = {"relax": 0.25, "talk": TALK, "look_down": 0.3, "raise1": RAISE, "rest": 0.28, "raise2": RAISE,
         "after": 0.27}


def test_steps_run_in_order_and_record_frames():
    steps = (Step("a", 1.0, "do a"), Step("b", 0.5, "do b"))
    run = CalibrationRun(10.0, steps)
    seen = []
    k = 0
    while (state := run.feed(Signals(10.0 + k * DT, 0, [], face(0.3)), 10.0 + k * DT)) is not None:
        seen.append(state[1].name)
        k += 1
    assert seen.count("a") == 30 and seen.count("b") == 15 and run.done
    assert [f.step for f in run.frames] == seen and run.frames[0].sig.mask is None
    assert run.frames[31].t == pytest.approx(DT, abs=1e-6)  # seconds into its own step


def test_default_steps_take_about_a_minute():
    assert 45.0 <= sum(s.seconds for s in cal.STEPS) <= 65.0
    names = [s.name for s in cal.STEPS]
    for g, steps in cal.POSITIVE.items():
        assert set(steps) <= set(names), g


def test_brows_fire_on_raises_and_not_while_talking():
    res = analyse(session(ORDER), positive=POS)
    v = res.verdicts[EYEBROWS_UP]
    assert v.ok and v.fired and v.false_s < 0.3, v
    assert res.brows["signal"] == "blend" and res.brows["rise"] == pytest.approx(0.6, abs=0.05)
    # replayed with the chosen setting: Jim in both raises, never in the talk step
    det = PoseDetector()
    det.apply_calibration({"brows": res.brows})
    on = {s: 0 for s in ORDER}
    for f in session(ORDER):
        on[f.step] += det.update(f.sig, f.sig.t)[EYEBROWS_UP]
    assert on["raise1"] > 20 and on["raise2"] > 20 and on["talk"] == 0 and on["relax"] == 0


def test_brows_that_rise_as_much_when_talking_are_not_saved():
    loud = {**ORDER, "talk": lambda s: 0.3 if s < 1.0 else 0.85}  # the brows go all the way up while talking
    v = analyse(session(loud), positive=POS)
    assert analyse(session(loud), positive=POS).brows is None  # the previous setting stays
    assert not v.verdicts[EYEBROWS_UP].ok and "raise them higher" in v.verdicts[EYEBROWS_UP].line()


def test_brows_that_never_rise_are_reported():
    flat = {k: 0.3 for k in ORDER}
    res = analyse(session(flat), positive=POS)
    assert res.brows is None and "NOT SEEN" in res.verdicts[EYEBROWS_UP].line()


def test_a_late_raise_still_counts():
    late = {**ORDER, "raise1": lambda s: 0.25 if s < 2.8 else 0.85, "rest": lambda s: 0.85 if s < 0.9 else 0.28}
    assert analyse(session(late), positive=POS).verdicts[EYEBROWS_UP].fired


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
    res = analyse(session(ORDER), positive=POS)
    app._save_calibration(res.calibration())
    loaded = app._load_calibration()
    assert loaded["version"] == cal.VERSION and loaded["brows"] == res.brows and "report" in loaded


# ---------------------------------------------------------------- a real user's recording
SITBACK = Path(__file__).parent / "fixtures_local" / "live_check_sitback.json"


def test_real_recording_forehead_collar_and_cover_check_out():
    """The user reacted ~2.5 s late to each instruction in this recording; shifted by that, Michael Scott,
    the collar and covering the face are all recognised in their own steps and nowhere else."""
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
