"""Calibration: each step waits for the action to be seen; the analysis tunes to the person; saving."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
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


def drive(run: Calibration, person: Person | None, t0: float = 0.0, seconds: float = 200.0, steps=None,
          delay: float = 0.0, away=lambda t: False):
    """Feed frames until the run is done (or ``seconds`` pass). ``person`` None = sits there doing nothing.
    The person does each step's action once its prompt shows (``delay`` seconds later), and lets go when it's
    ticked off; ``away(t)``: out of the picture."""
    det = PoseDetector()
    views = []
    shown = (-1, t0)
    k = 0
    t = t0
    while t < t0 + seconds:
        step = run.step.name if run.step is not None else "relax"
        if steps is not None and step not in steps:
            break
        if run.i != shown[0]:
            shown = (run.i, t)  # the prompt of a new step appears
        acting = not views or views[-1].state in ("read", "wait", "seen")
        doing = step if acting and t - shown[1] >= delay else "relax"
        f = person.face_for(doing, k) if person is not None else face(0.25)
        f = None if away(t) else f
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
    views = drive(run, None, seconds=40.0)
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



# ---------------------------------------------------------------- at your pace
def test_the_next_prompt_waits_until_you_let_go():
    run = Calibration(0.0, FACE_ONLY, lead_s=0.5)
    chatty = Person()
    chatty.face_for = lambda step, k: (face(0.25, jaw=0.05 + 0.3 * abs(math.sin(0.4 * k)))
                                       if step in ("talk", "relax") and k > 30 else Person.face_for(chatty, step, k))
    views = drive(run, chatty, seconds=18.0)  # keeps talking after "got it!"
    assert run.step.name == "talk" and views[-1].state == "ready" and "stop talking" in views[-1].title
    drive(run, chatty, t0=18.0, seconds=6.0, steps=("talk",))
    assert run.step.name == "look_down"  # ... but not forever

    run = Calibration(0.0, FACE_ONLY, lead_s=0.5)
    views = drive(run, Person(), steps=("relax", "talk"))  # stops when it's ticked off
    ready = [v for v in views if v.state == "ready"]
    assert run.step.name == "look_down" and len(ready) * DT < 0.5


def test_a_step_pauses_while_you_are_away_and_waits_for_you():
    run = Calibration(0.0, FACE_ONLY, lead_s=0.5)
    views = drive(run, None, seconds=60.0, away=lambda t: t > 9.0)  # leaves during the talk step
    assert run.step.name == "talk" and views[-1].state == "pause" and "paused" in views[-1].sub
    assert run._waited < 3.0  # the minute away doesn't count as waiting: no hint, no skip offer
    views = drive(run, Person(), t0=60.0, steps=("talk",))  # back, and talks
    assert run.step.name == "look_down"


def test_getting_ready_counts_only_while_you_are_in_the_picture():
    run = Calibration(0.0, FACE_ONLY, lead_s=2.0)
    views = drive(run, None, seconds=10.0, away=lambda t: True)
    assert views[-1].state == "lead" and "sit where you usually do" in views[-1].sub
    views = drive(run, None, t0=10.0, seconds=3.0)
    assert views[-1].state in ("read", "wait") and views[-1].title == cal.STEP_BY_NAME["relax"].prompt


def test_hints_wait_longer_for_people_who_take_their_time():
    run = Calibration(0.0, FACE_ONLY, lead_s=0.5)
    drive(run, Person(), delay=8.0)  # gets into each pose about 8 s after it's asked
    assert run.done and run.pace == pytest.approx(7.0, abs=0.3)  # (minus the second to read the prompt)
    assert run.retry([EYEBROWS_UP], 300.0)
    views = drive(run, None, t0=300.0, seconds=25.0)  # now doesn't raise them at all
    assert not any("N = skip" in v.sub for v in views)  # someone quicker is offered a skip after 20 s
    views = drive(run, None, t0=325.0, seconds=8.0)
    assert "N = skip" in views[-1].sub


def test_a_step_that_follows_carries_straight_on():
    relax = cal.STEP_BY_NAME["relax"]
    run = Calibration(0.0, (relax, replace(relax, name="again", follows=True)), lead_s=0.5)
    views = drive(run, Person())
    assert run.done
    after = [v.state for v in views if v.n == 2]
    assert after[0] in ("wait", "seen") and "read" not in after  # no tick, no prompt pause in between

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


# ---------------------------------------------------------------- your original calibration
def saved(when: str, ok: dict, params: dict | None = None, brows: dict | None = None) -> dict:
    return {"version": cal.VERSION, "when": when, "ok": ok, "params": params or {}, "brows": brows, "report": {}}


def test_your_first_calibration_is_kept_as_the_original(tmp_path, monkeypatch):
    from zap import app

    monkeypatch.setenv("ZAP_HOME", str(tmp_path))
    first, second = saved("2026-11-01 10:00", {FOREHEAD: True}), saved("2026-11-02 10:00", {FOREHEAD: True})
    app._save_calibration(first)
    app._save_calibration(second)
    folder = tmp_path / "calibration"
    assert json.loads((folder / "original.json").read_text())["when"] == "2026-11-01 10:00"
    assert app._load_calibration()["when"] == "2026-11-02 10:00"
    assert sorted(p.name for p in folder.iterdir()) == ["gestures.json", "original.json"]  # no temp files left
    (folder / "gestures.json").write_text("{half a fi")  # damaged: your original takes over
    assert app._load_calibration()["when"] == "2026-11-01 10:00"
    assert "2026-11-01" in app.restore_original_calibration()
    assert json.loads((folder / "gestures.json").read_text())["when"] == "2026-11-01 10:00"


def test_a_calibration_from_before_originals_were_kept_becomes_the_original(tmp_path, monkeypatch):
    from zap import app

    monkeypatch.setenv("ZAP_HOME", str(tmp_path))
    assert "No original" in app.restore_original_calibration()
    folder = tmp_path / "calibration"
    folder.mkdir()
    old = {"version": cal.VERSION, "when": "2026-10-30 02:25", "brows": {"signal": "blend", "rise": 0.386},
           "params": {}, "report": {EYEBROWS_UP: "Jim (eyebrows): OK", FOREHEAD: "Michael Scott (forehead): OK"}}
    (folder / "gestures.json").write_text(json.dumps(old))
    app._save_calibration(saved("2026-11-01 10:00", {}))
    assert json.loads((folder / "original.json").read_text()) == old
    assert cal.worked(old, EYEBROWS_UP) and not cal.worked(old, COLLAR)  # older files: read from the report


def test_what_failed_this_time_keeps_the_setting_that_worked_before():
    original = saved("2026-10-30", {EYEBROWS_UP: True, "peace": True, COLLAR: True},
                     brows={"signal": "geo", "rise": 0.02}, params={"peace": {"min_score": 0.35}})
    last = saved("2026-10-31", {EYEBROWS_UP: False, "peace": True, COLLAR: True}, brows=None)
    new = saved("2026-11-01", {EYEBROWS_UP: False, "peace": False, COLLAR: True, FOREHEAD: False},
                params={"peace": {"min_score": 0.5}, COLLAR: {"x": 1}})
    out = cal.keep_working(new, last, original)
    assert out["brows"] == {"signal": "geo", "rise": 0.02}  # Jim last worked in your original
    assert "peace" not in out["params"]  # it worked last time, with the shipped settings
    assert out["params"][COLLAR] == {"x": 1}  # worked this time: the new setting
    assert out["kept"] == {EYEBROWS_UP: "2026-10-30", "peace": "2026-10-31"}
    assert out["ok"][EYEBROWS_UP] and not out["ok"][FOREHEAD]  # Michael Scott never worked: nothing to keep
    assert new["params"]["peace"] == {"min_score": 0.5}  # the fresh result itself is left alone
    other_version = {**original, "version": cal.VERSION - 1}
    assert cal.keep_working(new, other_version)["kept"] == {}


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_calibration_files_live_in_each_systems_user_folder(platform, tmp_path, monkeypatch):
    from zap import app

    monkeypatch.delenv("ZAP_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    app._save_calibration(saved("2026-11-01 10:00", {FOREHEAD: True}))
    where = {"win32": tmp_path / "AppData" / "Roaming" / "ZAP", "darwin": tmp_path / "Library" / "Application Support" /
             "ZAP", "linux": tmp_path / ".config" / "ZAP"}[platform] / "calibration"
    assert (where / "gestures.json").is_file() and (where / "original.json").is_file()
    assert app._load_calibration()["when"] == "2026-11-01 10:00"


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


def test_the_app_saves_the_result_and_says_what_it_kept(tmp_path, monkeypatch):
    """The end of Ctrl+Alt+C in the app: a gesture that failed this time keeps the setting from your original."""
    from types import SimpleNamespace

    from zap import app

    monkeypatch.setenv("ZAP_HOME", str(tmp_path))
    app._save_calibration(saved("2026-10-30 02:25", {EYEBROWS_UP: True, FOREHEAD: True},
                                brows={"signal": "blend", "rise": 0.386}))
    run = Calibration(0.0, FACE_ONLY, lead_s=0.5)
    run.retried = True  # the one retry is used up
    verdicts = {EYEBROWS_UP: cal.Verdict(EYEBROWS_UP, False, 0.0), FOREHEAD: cal.Verdict(FOREHEAD, True, 0.0)}
    job = {"result": cal.Result(verdicts, None, {}), "error": None, "run": run}
    stub = SimpleNamespace(_calib=None, _calib_job=job, det=PoseDetector(), notices=app.Notices(), _banner=None,
                           opts=app.RunOptions(), _quit_at=None)
    stub._show_banner = lambda *a, **kw: setattr(stub, "_banner", a)
    app.App._calibrate_tick(stub, None, 10.0, {})
    now = app._load_calibration()
    assert now["kept"] == {EYEBROWS_UP: "2026-10-30 02:25"} and now["brows"]["rise"] == 0.386
    assert stub.det._brows.rise == 0.386  # in use straight away
    texts = [t for _, t in stub.notices.active()]
    assert any("kept your earlier setting: Jim" in t for t in texts)
    assert not any("NOT RECOGNISED" in t for t in texts)  # it has a working setting, so it isn't an error
    assert "1 kept from before" in stub._banner[1]
    original = json.loads((tmp_path / "calibration" / "original.json").read_text())
    assert original["when"] == "2026-10-30 02:25"  # your first one is still there
