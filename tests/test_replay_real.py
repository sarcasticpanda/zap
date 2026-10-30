"""Replays of real recordings of one user through PoseDetector + Arbiter (+ vision-only snaps).

The recordings live in tests/fixtures_local (gitignored, never published); the
tests skip when they are missing. Format: a list of per-frame dicts, see
zap.recording. Steps of live_check_sitback (seconds): setup 6, relax 5,
raise1/rest1/raise2/rest2/raise3 3 each, rest3 2, forehead 6, collar 6, cover 5,
uncover 3, snap_pose 6. The user reacted late to every instruction, so a pose
often shows up at the start of the following step.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest

from zap.gestures.snap import LOAD_MW, LOAD_TM, RELEASE_TM, snap_features
from zap.recording import ReplayReport, load_frames, replay_frames, signals_from_record

FIXTURES = Path(__file__).parent / "fixtures_local"
SITBACK = FIXTURES / "live_check_sitback.json"
CLOSE = FIXTURES / "live_check_close.json"

# In the sitback recording the forehead hand is first detected 5.0 s into the 6 s forehead
# step and stays on the forehead until 35.74 s (1.7 s into the collar step).
FOREHEAD_HAND_LEAVES = 35.74


def _need(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"local recording {path.name} not present")


@cache
def _report(name: str) -> ReplayReport:
    return replay_frames(load_frames(FIXTURES / name))


def _triggers(report: ReplayReport, gesture: str) -> list[tuple[float, str]]:
    return [(e.t, e.step) for e in report.events if e.kind == "trigger" and e.gesture == gesture]


def test_close_hands_up_fires_during_hands_up_step():
    _need(CLOSE)
    steps = [s for _, s in _triggers(_report(CLOSE.name), "hands_up")]
    assert steps and set(steps) <= {"hands", "hands_up"}  # the user raised both hands from the "hands" step on


def test_nothing_fires_while_just_sitting():
    for path, quiet in ((CLOSE, {"look"}), (SITBACK, {"setup", "relax"})):
        _need(path)
        fired = [e for e in _report(path.name).events if e.step in quiet]
        assert fired == [], fired


def test_sitback_eyebrows_fire_in_the_brow_steps_only():
    _need(SITBACK)
    steps = [s for _, s in _triggers(_report(SITBACK.name), "eyebrows_up")]
    assert any(s in {"raise1", "rest1", "raise2", "rest2", "raise3", "rest3"} for s in steps)
    assert not set(steps) & {"forehead", "collar", "cover", "uncover", "snap_pose"}


def test_sitback_collar_fires_during_collar_step():
    _need(SITBACK)
    assert "collar" in [s for _, s in _triggers(_report(SITBACK.name), "collar")]


def test_sitback_cover_face_fires_when_the_hands_drop():
    _need(SITBACK)
    steps = [s for _, s in _triggers(_report(SITBACK.name), "cover_face")]
    assert steps and set(steps) <= {"cover", "uncover"}


def test_sitback_forehead_fires_while_the_hand_is_on_the_forehead():
    _need(SITBACK)
    fired = _triggers(_report(SITBACK.name), "forehead")
    frames = load_frames(SITBACK)
    start = next(f["t"] for f in frames if f["step"] == "forehead")
    assert len(fired) == 1
    t, _ = fired[0]
    assert start <= t < FOREHEAD_HAND_LEAVES
    collar_t = [t for t, _ in _triggers(_report(SITBACK.name), "collar")]
    assert collar_t and collar_t[0] > t


def test_sitback_snap_pose_loaded_and_released_frames():
    """The user loaded the snap, snapped (holding the finger in the palm for ~2.5 s), then loaded again."""
    _need(SITBACK)
    loaded = released = total = 0
    for i, rec in enumerate(load_frames(SITBACK)):
        if rec["step"] != "snap_pose":
            continue
        for h in signals_from_record(rec, i).hands:
            _, tm, mw = snap_features(h)
            total += 1
            loaded += tm < LOAD_TM and mw > LOAD_MW
            released += tm > RELEASE_TM and mw < LOAD_MW - 0.18
    assert total > 50
    assert loaded >= 0.3 * total  # clearly loaded: tm 0.04-0.17, mw 1.27-1.51
    assert loaded + released >= 0.8 * total  # nearly every frame is one or the other


def test_sitback_vision_only_snap_detected_once():
    _need(SITBACK)
    report = _report(SITBACK.name)
    snaps = [e for e in report.events if e.kind == "snap"]
    assert len(snaps) == 1 and snaps[0].step == "snap_pose"
    assert [s for _, s in _triggers(report, "snap1")] == ["snap_pose"]


def test_replay_is_deterministic():
    _need(CLOSE)
    frames = load_frames(CLOSE)[:300]
    a, b = replay_frames(frames), replay_frames(frames)
    assert [(e.t, e.kind, e.gesture) for e in a.events] == [(e.t, e.kind, e.gesture) for e in b.events]


def test_signals_round_trip_matches_fixture_shapes():
    _need(CLOSE)
    rec = next(r for r in load_frames(CLOSE) if r["face"] and r["hands"])
    sig = signals_from_record(rec, 7)
    assert sig.face.lm.shape == (478, 3) and np.all(sig.face.lm[:, 2] == 0)
    assert sig.hands[0].lm.shape == (21, 3) and sig.width == 1280
