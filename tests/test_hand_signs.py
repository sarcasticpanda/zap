"""One-hand signs (one finger up, thumbs down, rock on): detection, rejection, arbitration, bindings.

Hands are built from landmark layouts in hand units (wrist at the origin, image
y down) so each sign has its real finger shape, not just a label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zap import config as zap_config
from zap.clock import ReplayClock
from zap.gestures.arbiter import Arbiter
from zap.gestures.poses import PoseDetector, point_up_shape, rock_shape, thumb_down_shape, thumb_up_shape
from zap.gestures.types import HAND_SIGNS, POINT_UP, ROCK, THUMBS_DOWN
from zap.signals import Face, Hand, Signals
from zap.ui.preview import pose_rows

W, H = 1280, 720
DT = 1 / 30
FW, CX, CY = 200.0, 640.0, 300.0  # face width and nose bridge in pixels
S = 0.7 * FW  # hand size (wrist to middle knuckle)

# wrist, thumb cmc mcp ip tip, then index / middle / ring / pinky mcp pip dip tip
_THUMB_TUCKED = [(-0.3, -0.2), (-0.45, -0.4), (-0.4, -0.6), (-0.2, -0.7)]
_THUMB_OUT = [(-0.3, -0.2), (-0.5, -0.4), (-0.65, -0.6), (-0.8, -0.75)]
_INDEX_UP = [(-0.25, -0.95), (-0.27, -1.35), (-0.29, -1.6), (-0.3, -1.85)]
_MIDDLE_UP = [(0.0, -1.0), (0.0, -1.45), (0.0, -1.72), (0.0, -1.95)]
_MIDDLE_CURLED = [(0.0, -1.0), (0.02, -1.3), (0.05, -1.0), (0.03, -0.8)]
_RING_CURLED = [(0.22, -0.95), (0.25, -1.2), (0.26, -0.95), (0.24, -0.78)]
_PINKY_UP = [(0.42, -0.85), (0.48, -1.15), (0.52, -1.35), (0.55, -1.5)]
_PINKY_CURLED = [(0.42, -0.85), (0.46, -1.05), (0.46, -0.85), (0.44, -0.72)]
# a fist on its side with the thumb up (fingers curled toward image-left)
_THUMBS_UP = [(0.0, 0.0),
              (-0.3, -0.25), (-0.55, -0.55), (-0.62, -0.85), (-0.65, -1.15),
              (-0.9, -0.45), (-1.2, -0.45), (-1.1, -0.3), (-0.9, -0.3),
              (-0.95, -0.2), (-1.25, -0.2), (-1.15, -0.05), (-0.95, -0.05),
              (-0.9, 0.0), (-1.18, 0.02), (-1.08, 0.15), (-0.9, 0.15),
              (-0.8, 0.18), (-1.02, 0.22), (-0.95, 0.32), (-0.8, 0.32)]

SHAPES = {
    POINT_UP: [(0.0, 0.0), *_THUMB_TUCKED, *_INDEX_UP, *_MIDDLE_CURLED, *_RING_CURLED, *_PINKY_CURLED],
    ROCK: [(0.0, 0.0), *_THUMB_OUT, *_INDEX_UP, *_MIDDLE_CURLED, *_RING_CURLED, *_PINKY_UP],
    THUMBS_DOWN: [(x, -y) for x, y in _THUMBS_UP],
}
OPEN = [(0.0, 0.0), *_THUMB_OUT, *_INDEX_UP, *_MIDDLE_UP, (0.22, -0.95), (0.25, -1.35), (0.27, -1.6), (0.28, -1.8),
        *_PINKY_UP]
LABEL = {POINT_UP: "Pointing_Up", THUMBS_DOWN: "Thumb_Down", ROCK: "ILoveYou"}


def px(u: float, v: float) -> np.ndarray:
    return np.array([CX + u * FW, CY + v * FW])


def make_face() -> Face:
    n = 478
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.sqrt(np.linspace(0.05, 1.0, n))
    u, v = 0.48 * r * np.cos(ang), 0.14 + 0.62 * r * np.sin(ang)
    key = {10: (0.0, -0.5), 168: (0.0, 0.0), 1: (0.0, 0.33), 152: (0.0, 0.78), 234: (-0.5, 0.05),
           454: (0.5, 0.05), 105: (-0.25, -0.12), 334: (0.25, -0.12), 145: (-0.2, 0.08), 374: (0.2, 0.08)}
    for i, (uu, vv) in key.items():
        u[i], v[i] = uu, vv
    lm = np.zeros((n, 3), np.float32)
    lm[:, 0], lm[:, 1] = CX + u * FW, CY + v * FW
    return Face(lm, {}, None, 0.0, True)


def sign_hand(name_or_pts, wrist, label: str = "None", score: float = 0.0) -> Hand:
    pts = SHAPES[name_or_pts] if isinstance(name_or_pts, str) else name_or_pts
    lm = np.zeros((21, 3), np.float32)
    lm[:, :2] = np.asarray(wrist, float) + np.asarray(pts, float) * S
    return Hand(lm, 0.95, label, score)


def run(det: PoseDetector, hands_per_frame, face=None, t0: float = 0.0) -> list[dict[str, bool]]:
    out = []
    for k, hands in enumerate(hands_per_frame):
        t = t0 + k * DT
        out.append(det.update(Signals(t, k, list(hands), face, width=W, height=H), t))
    return out


BESIDE = px(1.7, 1.2)  # wrist beside the face, below the cheek: the sign is held up next to the head


# ---------------------------------------------------------------- shapes
@pytest.mark.parametrize("name", HAND_SIGNS)
def test_each_shape_is_recognised_and_only_its_own(name):
    checks = {POINT_UP: point_up_shape, THUMBS_DOWN: thumb_down_shape, ROCK: rock_shape}
    h = sign_hand(name, BESIDE)
    assert {n for n, f in checks.items() if f(h)} == {name}
    assert not thumb_up_shape(h)


def test_open_hand_and_thumbs_up_are_no_sign():
    for pts in (OPEN, _THUMBS_UP):
        h = sign_hand(pts, BESIDE)
        assert not (point_up_shape(h) or thumb_down_shape(h) or rock_shape(h))


def test_rock_without_the_thumb_counts_too():
    pts = [(0.0, 0.0), *_THUMB_TUCKED, *_INDEX_UP, *_MIDDLE_CURLED, *_RING_CURLED, *_PINKY_UP]  # 🤘
    assert rock_shape(sign_hand(pts, BESIDE))


def test_a_finger_pointing_sideways_is_not_one_finger_up():
    pts = [(y, x) for x, y in SHAPES[POINT_UP]]  # rotated 90 degrees: pointing at image-left
    assert not point_up_shape(sign_hand(pts, BESIDE))


# ---------------------------------------------------------------- detector
@pytest.mark.parametrize("name", HAND_SIGNS)
def test_sign_detected_beside_the_face(name):
    res = run(PoseDetector(), [[sign_hand(name, BESIDE, LABEL[name], 0.7)]] * 10, make_face())
    assert all(r[name] for r in res)
    assert not any(r[other] for r in res for other in HAND_SIGNS if other != name)


@pytest.mark.parametrize("name", HAND_SIGNS)
def test_sign_detected_without_a_face(name):
    assert all(r[name] for r in run(PoseDetector(), [[sign_hand(name, BESIDE, LABEL[name], 0.7)]] * 5))


@pytest.mark.parametrize("name", HAND_SIGNS)
def test_label_needs_the_shape_unless_sure(name):
    wrong = OPEN  # the classifier says the sign, the fingers say open hand
    assert not any(r[name] for r in run(PoseDetector(), [[sign_hand(wrong, BESIDE, LABEL[name], 0.7)]] * 5))
    assert all(r[name] for r in run(PoseDetector(), [[sign_hand(wrong, BESIDE, LABEL[name], 0.85)]] * 5))


@pytest.mark.parametrize("name", [POINT_UP, THUMBS_DOWN])
def test_shape_alone_is_not_enough(name):
    # in real recordings a hand loaded for a snap looked like one finger up 16 times
    for label, score in (("None", 0.0), (LABEL[name], 0.4), ("Open_Palm", 0.9)):
        assert not any(r[name] for r in run(PoseDetector(), [[sign_hand(name, BESIDE, label, score)]] * 5))


def test_rock_shape_counts_while_the_classifier_sees_no_other_sign():
    # MediaPipe has no class for 🤘 (only 🤟): it says "None", so the shape decides
    for label, score in (("None", 0.9), ("ILoveYou", 0.3)):
        assert all(r[ROCK] for r in run(PoseDetector(), [[sign_hand(ROCK, BESIDE, label, score)]] * 5))
    for label in ("Victory", "Open_Palm", "Pointing_Up"):
        assert not any(r[ROCK] for r in run(PoseDetector(), [[sign_hand(ROCK, BESIDE, label, 0.7)]] * 5))


def test_finger_on_the_face_is_not_a_sign():
    # chin on the hand with the index finger up the cheek, or a finger at the lips ("shh")
    for wrist in (px(0.25, 1.35), px(0.05, 1.5)):
        h = sign_hand(POINT_UP, wrist, "Pointing_Up", 0.9)
        assert not any(r[POINT_UP] for r in run(PoseDetector(), [[h]] * 8, make_face()))


def test_sign_held_right_beside_the_cheek_counts():
    pts = [(-x, y) for x, y in SHAPES[POINT_UP]]  # the other hand: thumb toward the face
    h = sign_hand(pts, (730.0, 520.0), "Pointing_Up", 0.8)  # its box overlaps the face's by ~40 %
    assert all(r[POINT_UP] for r in run(PoseDetector(), [[h]] * 8, make_face()))


def test_hud_says_why_a_sign_does_not_count():
    det = PoseDetector()
    run(det, [[sign_hand(POINT_UP, px(0.25, 1.35), "Pointing_Up", 0.9)]] * 3, make_face())
    assert det.debug["sign_why"] == {POINT_UP: "on your face"}
    run(det, [[sign_hand(OPEN, BESIDE, "Thumb_Down", 0.6)]] * 3, make_face(), t0=1.0)
    assert det.debug["sign_why"] == {THUMBS_DOWN: "fingers?"}
    frames = [[sign_hand(ROCK, BESIDE + (k * 0.07 * FW, 0.0), "ILoveYou", 0.9)] for k in range(10)]
    run(det, frames, make_face(), t0=2.0)
    assert det.debug["sign_why"] == {ROCK: "hold still"}
    run(det, [[sign_hand(ROCK, BESIDE, "ILoveYou", 0.9)]] * 10, make_face(), t0=3.0)
    assert det.debug["sign_why"] == {}


def test_thumbs_down_over_the_face_is_ignored():
    h = sign_hand(THUMBS_DOWN, px(0.0, -0.6), "Thumb_Down", 0.9)
    assert not any(r[THUMBS_DOWN] for r in run(PoseDetector(), [[h]] * 8, make_face()))


def test_moving_hand_is_not_a_sign():
    frames = [[sign_hand(POINT_UP, BESIDE + (k * 0.07 * FW, 0.0), "Pointing_Up", 0.9)] for k in range(15)]
    res = run(PoseDetector(), frames, make_face())  # ~2 face widths per second
    assert not any(r[POINT_UP] for r in res[8:])


def test_flicker_to_none_is_tolerated_by_the_arbiter():
    cfg = _cfg()
    clock = ReplayClock(0.0)
    arb, det = Arbiter(cfg, clock), PoseDetector()
    fired = []
    for k in range(60):
        t = k * DT
        clock.set(t)
        label = "None" if k % 9 == 4 else "Thumb_Down"  # one frame in nine the classifier loses it
        truths = det.update(Signals(t, k, [sign_hand(THUMBS_DOWN, BESIDE, label, 0.8)], make_face(), width=W, height=H), t)
        fired += [i.gesture for i in arb.update(truths, (), t, t_on=det.t_on) if i.kind == "trigger"]
    assert fired == [THUMBS_DOWN]


# ---------------------------------------------------------------- arbiter
def _cfg():
    cfg = zap_config.defaults()
    cfg.app.armed_on_start = True
    cfg.app.arm_grace_s = 0.0
    cfg.zoom.leave_guard_s = 0.0
    return cfg


def _hold(name, seconds, busy=False):
    cfg = _cfg()
    clock = ReplayClock(0.0)
    arb, det = Arbiter(cfg, clock), PoseDetector()
    out = []
    for k in range(round(seconds / DT)):
        t = k * DT
        clock.set(t)
        truths = det.update(Signals(t, k, [sign_hand(name, BESIDE, LABEL[name], 0.8)], make_face(), width=W, height=H), t)
        out += [(round(i.t, 2), i.gesture) for i in arb.update(truths, (), t, busy=busy, t_on=det.t_on)
                if i.kind == "trigger"]
    return out


@pytest.mark.parametrize("name", HAND_SIGNS)
def test_held_sign_fires_once(name):
    fired = _hold(name, 3.0)
    assert [g for _, g in fired] == [name]
    assert 0.6 <= fired[0][0] <= 1.0  # about a second of holding, no more


def test_quick_raise_of_one_finger_does_not_fire():
    assert _hold(POINT_UP, 0.5) == []


def test_sign_waits_while_a_meme_is_showing():
    assert _hold(ROCK, 2.0, busy=True) == []


# ---------------------------------------------------------------- bindings and HUD
def test_default_bindings_give_every_effect_its_own_sign():
    b = zap_config.defaults().bindings
    assert b[POINT_UP] == "brb"
    assert b[THUMBS_DOWN] == "cutaway:michael_scott"
    assert b[ROCK] == "cutaway:jim"
    assert b["forehead"] == "none" and b["eyebrows_up"] == "none"  # they fired by accident
    held = [g for g in b if not g.startswith("snap") and b[g] != "none"]
    targets = [b[g] for g in held if b[g] != "outro"]  # the outro also has snap x2 and peace
    assert len(targets) == len(set(targets)), "two signs for one meme"


def test_hud_rows_follow_the_bindings():
    rows = dict(pose_rows(zap_config.defaults().bindings))
    assert rows[POINT_UP] == "1 finger up > right back"
    assert rows[THUMBS_DOWN] == "thumbs down > michael"
    assert rows[ROCK] == "rock on > jim"
    assert "forehead" not in rows and "eyebrows_up" not in rows


def test_brb_config_section(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[brb]\ntext = "BRB"\nseconds = 3\n[bindings]\nrock = "brb"\n', encoding="utf-8")
    cfg = zap_config.load(p)
    assert cfg.brb.text == "BRB" and cfg.brb.seconds == 3.0 and cfg.brb.hold_when_away
    assert cfg.bindings[ROCK] == "brb" and cfg.bindings[POINT_UP] == "brb"
    p.write_text("[brb]\nhold_when_awy = true\n", encoding="utf-8")
    with pytest.raises(zap_config.ConfigError, match="did you mean 'hold_when_away'"):
        zap_config.load(p)


# ---------------------------------------------------------------- real recordings
FIXTURES = Path(__file__).parent / "fixtures_local"


@pytest.mark.parametrize("name", ["live_check_close.json", "live_check_sitback.json"])
def test_no_sign_fires_in_real_recordings(name):
    """A real user's hands up, forehead, collar, cover-face and snap poses never read as a sign."""
    from zap.recording import load_frames, replay_frames

    path = FIXTURES / name
    if not path.is_file():
        pytest.skip(f"local recording {name} not present")
    fired = [e for e in replay_frames(load_frames(path)).events if e.gesture in HAND_SIGNS]
    assert fired == []
