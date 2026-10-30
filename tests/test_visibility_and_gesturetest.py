"""The 'can ZAP see you' status, and the zap test flow (setup, quiet, gestures, report)."""

from __future__ import annotations

import json

import numpy as np

from zap.gestures import gesturetest
from zap.gestures.gesturetest import GestureStep, GestureTest
from zap.gestures.poses import PoseDetector
from zap.gestures.types import PEACE, TWO_THUMBS, Intent
from zap.signals import Face, Hand, Signals
from zap.vision.visibility import Visibility

W, H = 1280, 720
DT = 1 / 30


def face(chin_y: float = 470.0, width: float = 200.0, cx: float = 640.0) -> Face:
    lm = np.zeros((478, 3), np.float32)
    lm[:, 0], lm[:, 1] = cx, chin_y - 150
    lm[234, :2], lm[454, :2] = (cx - width / 2, chin_y - 160), (cx + width / 2, chin_y - 160)
    lm[152, :2] = (cx, chin_y)
    lm[168, :2] = (cx, chin_y - 180)
    return Face(lm, {}, None, 0.0, True)


def hand() -> Hand:
    lm = np.zeros((21, 3), np.float32)
    lm[:, 0], lm[:, 1] = np.linspace(850, 950, 21), np.linspace(450, 300, 21)
    lm[9, :2] = (900, 360)
    return Hand(lm, 0.9, "None", 0.0)


def sig(t: float, f: Face | None = None, hands=(), luma: float = 120.0, face_luma: float | None = 120.0) -> Signals:
    return Signals(t, 0, list(hands), f, None, False, W, H, luma, face_luma if f is not None else None)


# ---------------------------------------------------------------- visibility
def test_all_good_has_no_problems():
    v = Visibility()
    v.update(sig(0.0, face()), 0.0)
    assert v.problems(0.0) == [] and v.face_ok(0.0)


def test_no_face_says_so_and_why():
    v = Visibility()
    v.update(sig(0.0, None), 0.0)
    assert "can't see your face" in v.problems(2.0)[0][1]
    for k in range(40):  # brightness is smoothed over a second or so
        v.update(sig(0.1 + k * DT, None, luma=20.0), 0.1 + k * DT)
    assert "too dark" in v.problems(2.0)[0][1]


def test_low_framing_is_an_error():
    v = Visibility()
    v.update(sig(0.0, face(chin_y=680.0)), 0.0)  # 40 px below the chin = 0.2 face widths
    sev, text = v.problems(0.0)[0]
    assert sev == "error" and "shoulders" in text


def test_backlight_is_flagged():
    v = Visibility()
    v.update(sig(0.0, face(), luma=160.0, face_luma=50.0), 0.0)
    assert any("behind you" in t for _, t in v.problems(0.0))


# ---------------------------------------------------------------- the gesture test
STEPS = (
    GestureStep("setup", "setup", None, 10.0, "setup"),
    GestureStep("quiet", "quiet", None, 2.0, "quiet"),
    GestureStep("peace", "peace", PEACE, 3.0),
    GestureStep("thumbs", "thumbs", TWO_THUMBS, 3.0),
)


def run_test(script) -> GestureTest:
    """script(step_name, t_in_step) -> (signals_kwargs, intents)"""
    test = GestureTest(0.0, STEPS)
    det, vis = PoseDetector(), Visibility()
    t, k = 0.0, 0
    started = {}
    while not test.done and t < 60:
        name = test.step.name if test.step else ""
        started.setdefault((name, test.i), t)
        kw, intents = script(name, t - started[(name, test.i)])
        s = sig(t, **kw)
        vis.update(s, t)
        test.feed(s, det, {}, intents, vis, t)
        k += 1
        t = k * DT
    return test


def test_flow_passes_and_fails_with_reasons():
    def script(name, t):
        if name == "setup":
            return {"f": face(), "hands": [hand()]}, []
        if name == "quiet":
            return {"f": face()}, []
        if name == "peace":
            return {"f": face(), "hands": [hand()]}, ([Intent("trigger", PEACE, t)] if t > 1.0 else [])
        return {"f": face()}, []  # thumbs: the hand never shows

    test = run_test(script)
    r = {x.name: x for x in test.results}
    assert r["setup"].passed and r["quiet"].passed
    assert r["peace"].passed and 1.0 <= r["peace"].seconds <= 1.2
    assert not r["thumbs"].passed and "no hand seen" in r["thumbs"].detail


def test_quiet_step_catches_false_triggers():
    def script(name, t):
        if name == "quiet":
            return {"f": face()}, ([Intent("trigger", "eyebrows_up", t)] if 0.5 < t < 0.55 else [])
        return {"f": face(), "hands": [hand()]}, ([Intent("trigger", name if name != "thumbs" else TWO_THUMBS, t)]
                                                 if name in ("peace", "thumbs") else [])

    test = run_test(script)
    quiet = next(x for x in test.results if x.name == "quiet")
    assert not quiet.passed and "eyebrows_up" in quiet.detail


def test_setup_fails_on_bad_framing():
    test = run_test(lambda name, t: ({"f": face(chin_y=690.0), "hands": [hand()]}, []))
    setup = test.results[0]
    assert not setup.passed and "shoulders" in setup.detail


def test_save_writes_report_and_landmarks(tmp_path):
    test = run_test(lambda name, t: ({"f": face(), "hands": [hand()]}, []))
    folder = gesturetest.save(test, tmp_path / "t")
    report = json.loads((folder / "report.json").read_text())
    assert report["total"] == len(STEPS) and (folder / "signals.jsonl").stat().st_size > 0
    assert "Gesture test:" in gesturetest.format_report(test)
