"""SnapFusion on synthetic hand sequences (no camera)."""

from __future__ import annotations

import numpy as np
import pytest

from zap.gestures import snap as sn
from zap.gestures.snap import BurstCounter, SnapFusion, snap_features
from zap.gestures.snap_audio import AudioCandidate
from zap.signals import Hand, Signals

FPS = 30.0
LOADED = (0.20, 1.30)  # (tm, mw)
RELEASED = (0.60, 1.00)
STRICT = (0.70, 0.85)  # a real flick: the middle tip lands near the palm
NEUTRAL = (0.90, 1.60)  # open hand, middle finger up, thumb away


def hand(cx=640.0, cy=360.0, s=100.0, tm=0.9, mw=1.6, score=0.9) -> Hand:
    lm = np.zeros((21, 3), np.float32)
    lm[:, :2] = (cx, cy - 0.5 * s)
    wrist = np.array([cx, cy + 0.5 * s])
    lm[0, :2] = wrist
    lm[9, :2] = (cx, cy - 0.5 * s)
    lm[5, :2] = (cx - 0.3 * s, cy - 0.45 * s)
    lm[13, :2] = (cx + 0.2 * s, cy - 0.45 * s)
    lm[17, :2] = (cx + 0.35 * s, cy - 0.35 * s)
    lm[12, :2] = wrist + (0.0, -mw * s)
    lm[4, :2] = lm[12, :2] + (tm * s, 0.0)
    return Hand(lm, score)


def frames(fusion, t0, t1, make):
    """Feed frames at 30 fps; make(t) -> list[Hand]."""
    for k in range(round((t1 - t0) * FPS)):
        t = t0 + k / FPS
        fusion.add_frame(Signals(t=t, seq=k, hands=make(t)))


def cand(t, accepted=True):
    return AudioCandidate(t, {}, accepted, "ok" if accepted else "LF heavy")


def snap_seq(t_release, x=640.0):
    def make(t):
        return [hand(x, tm=LOADED[0], mw=LOADED[1])] if t < t_release else [hand(x, tm=RELEASED[0], mw=RELEASED[1])]

    return make


def test_features_from_raw_landmarks():
    s, tm, mw = snap_features(hand(tm=0.25, mw=1.2))
    assert s == pytest.approx(100.0)
    assert tm == pytest.approx(0.25, abs=1e-4)
    assert mw == pytest.approx(1.2, abs=1e-4)


def test_loaded_then_release_accepted():
    f = SnapFusion()
    f.add_candidate(cand(1.0))
    frames(f, 0.0, 2.0, snap_seq(1.09))
    out = f.resolve(now=2.0)
    assert len(out) == 1
    t, where = out[0]
    assert t == 1.0
    assert where is not None and abs(where[0] - 640) < 50
    assert f.pending_onset is None


def test_release_only_counts_after_two_loaded_frames():
    f = SnapFusion()
    # A single loaded frame (noise) followed by a release is not a snap.
    frames(f, 0.0, 2.0, lambda t: [hand(tm=LOADED[0], mw=LOADED[1])] if abs(t - 1.0) < 0.01 else [hand(tm=0.6, mw=1.0)])
    f.add_candidate(cand(0.95))
    assert f.resolve(now=3.0) == []
    assert list(f.rejects) == [sn.NO_POSE]


def test_no_hands_rejected():
    f = SnapFusion()
    f.add_candidate(cand(1.0))
    frames(f, 0.0, 2.0, lambda t: [])
    assert f.resolve(now=2.0) == []
    assert list(f.rejects) == [sn.NO_HAND]


def test_unusable_hand_rejected():
    f = SnapFusion()
    f.add_candidate(cand(1.0))
    frames(f, 0.0, 2.0, lambda t: [hand(s=30.0, tm=LOADED[0], mw=LOADED[1])])  # too small: far away / speaker
    assert f.resolve(now=2.0) == []
    assert f.rejects[-1] == sn.NO_HAND


def test_clap_rejected():
    f = SnapFusion()
    f.add_candidate(cand(1.0))
    frames(f, 0.0, 2.0, lambda t: [hand(600, tm=LOADED[0], mw=LOADED[1]), hand(660)])
    assert f.resolve(now=2.0) == []
    assert f.rejects[-1] == sn.CLAP


def test_open_hand_without_pose_rejected():
    f = SnapFusion()
    f.add_candidate(cand(1.0))
    frames(f, 0.0, 2.0, lambda t: [hand(tm=NEUTRAL[0], mw=NEUTRAL[1])])
    assert f.resolve(now=2.0) == []
    assert f.rejects[-1] == sn.NO_POSE


def test_loaded_hand_losing_tracking_accepted():
    f = SnapFusion()
    f.add_candidate(cand(1.0))
    # Hand loaded, then gone from 1.09 (motion blur) and back open later.
    frames(f, 0.0, 2.0, lambda t: [hand(tm=LOADED[0], mw=LOADED[1])] if t < 1.09 else ([] if t < 1.3 else [hand()]))
    out = f.resolve(now=2.0)
    assert len(out) == 1


def test_waits_for_video_then_times_out():
    f = SnapFusion()
    f.add_candidate(cand(1.0))
    frames(f, 0.0, 1.2, snap_seq(1.09))  # covers only up to ~1.17 < tv + 0.25 = 1.33
    assert f.resolve(now=1.25) == []
    assert f.pending_onset == 1.0
    out = f.resolve(now=1.0 + 0.08 + 0.25 + 0.31)  # timeout: decide with what we have
    assert len(out) == 1
    assert f.pending_onset is None


def test_rejected_audio_ignored():
    f = SnapFusion()
    f.add_candidate(cand(1.0, accepted=False))
    frames(f, 0.0, 2.0, snap_seq(1.09))
    assert f.resolve(now=2.0) == []
    assert f.pending_onset is None and not f.rejects


def test_second_snap_of_burst_accepted_by_same_hand():
    f = SnapFusion()
    f.add_candidate(cand(1.0))
    f.add_candidate(cand(1.55))

    def make(t):
        if t < 1.09:
            return [hand(tm=LOADED[0], mw=LOADED[1])]
        return [hand(tm=RELEASED[0] + 0.05, mw=1.25)]  # no clean reload visible

    frames(f, 0.0, 2.0, make)
    out = f.resolve(now=2.0)
    assert [t for t, _ in out] == [1.0, 1.55]
    # Without the burst hint the second one has no pose.
    g = SnapFusion()
    g.add_candidate(cand(1.55))
    frames(g, 0.0, 2.0, make)
    assert g.resolve(now=2.0, in_burst=False) == []


def test_triple_snap_with_reloads():
    f = SnapFusion()
    for t in (1.0, 1.3, 1.6):
        f.add_candidate(cand(t))

    def make(t):
        phase = (t - 0.7) % 0.3
        loaded = t < 0.7 or phase < 0.2 or t > 1.7
        tm, mw = LOADED if loaded and t < 1.7 else RELEASED
        return [hand(tm=tm, mw=mw)]

    frames(f, 0.0, 2.5, make)
    assert [t for t, _ in f.resolve(now=2.5)] == [1.0, 1.3, 1.6]


def test_tracking_follows_moving_hand():
    f = SnapFusion()
    f.add_candidate(cand(1.0))

    def make(t):
        tm, mw = LOADED if t < 1.09 else RELEASED
        return [hand(400 + 60 * t, tm=tm, mw=mw)]

    frames(f, 0.0, 2.0, make)
    out = f.resolve(now=2.0)
    assert len(out) == 1 and out[0][1][0] > 400


def test_vision_only_strict_release():
    f = SnapFusion(vision_only=True)
    f.add_candidate(cand(1.0))  # ignored in vision-only mode

    def make(t):
        return [hand(tm=LOADED[0], mw=LOADED[1])] if t < 1.0 else [hand(tm=STRICT[0], mw=STRICT[1])]

    frames(f, 0.0, 1.5, make)
    out = f.resolve(now=1.5)
    assert len(out) == 1
    assert 0.85 < out[0][0] < 1.0
    assert f.resolve(now=1.6) == []


def test_vision_only_needs_strict_release():
    f = SnapFusion(vision_only=True)
    frames(f, 0.0, 1.5, snap_seq(1.0))  # the loose release is not enough without audio
    assert f.resolve(now=1.5) == []


def test_vision_counts_a_double_snap():
    """Seen snaps count up to 3 (two snaps = vanish and leave), no mic needed."""
    f = SnapFusion(vision_only=True)
    b = BurstCounter(gap_s=0.8)

    def make(t):
        loaded = t < 1.0 or 1.35 <= t < 1.6
        return [hand(tm=LOADED[0], mw=LOADED[1])] if loaded else [hand(tm=STRICT[0], mw=STRICT[1])]

    bursts = []
    for k in range(round(3.0 * FPS)):
        t = k / FPS
        f.add_frame(Signals(t=t, seq=k, hands=make(t)))
        for onset, where in f.resolve(now=t):
            b.add(onset, where)
        bursts += b.tick(t)[0]
    assert [x.count for x in bursts] == [2]


def test_vision_ignores_a_slow_pinch_release():
    """Thumb and middle finger drifting apart without the flick is not a snap."""
    f = SnapFusion(vision_only=True)

    def make(t):
        if t < 1.0:
            return [hand(tm=LOADED[0], mw=LOADED[1])]
        return [hand(tm=0.70, mw=1.25)]  # opened, but the middle finger stayed up

    frames(f, 0.0, 1.5, make)
    assert f.resolve(now=1.5) == []


def test_vision_ignores_a_two_hand_pinch_release():
    """Letting go of the collar with both hands pinched is not a snap, even with a snap-like flick."""
    f = SnapFusion(vision_only=True)

    def make(t):
        pose = LOADED if t < 1.0 else STRICT
        return [hand(cx=500.0, tm=pose[0], mw=pose[1]), hand(cx=800.0, tm=pose[0], mw=pose[1])]

    frames(f, 0.0, 1.5, make)
    assert f.resolve(now=1.5) == []


def test_burst_closes_early_when_the_hand_goes_down():
    b = BurstCounter(gap_s=1.3, quiet_s=0.5)
    b.add(1.0)
    assert b.tick(1.6, hold=True) == ([], [])  # hand still up: a second snap may come
    done, _ = b.tick(1.6, hold=False)  # hand lowered: one snap it is
    assert [x.count for x in done] == [1]
