"""One-hand signs (peace = We'll Be Right Back, L = To Be Continued): detection, rejection, arbitration,
bindings. Hands are built from landmark layouts in hand units (wrist at the origin, image y down) so each
sign has its real finger shape, not just a label."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zap import config as zap_config
from zap.clock import ReplayClock
from zap.gestures.arbiter import Arbiter
from zap.gestures.poses import PoseDetector, l_shape, peace_shape, peace_shape_strict, thumb_up_shape
from zap.gestures.types import HAND_SIGNS, L_SHAPE, PEACE
from zap.signals import Face, Hand, Signals
from zap.ui.preview import pose_rows

W, H = 1280, 720
DT = 1 / 30
FW, CX, CY = 200.0, 640.0, 300.0  # face width and nose bridge in pixels
S = 0.7 * FW  # hand size (wrist to middle knuckle)

# wrist, thumb cmc mcp ip tip, then index / middle / ring / pinky mcp pip dip tip
_THUMB_TUCKED = [(-0.3, -0.2), (-0.45, -0.4), (-0.4, -0.6), (-0.2, -0.7)]
_THUMB_OUT = [(-0.3, -0.2), (-0.6, -0.35), (-0.9, -0.45), (-1.2, -0.5)]  # sideways, like an L
_INDEX_UP = [(-0.25, -0.95), (-0.27, -1.35), (-0.29, -1.6), (-0.3, -1.85)]
_MIDDLE_UP = [(0.0, -1.0), (0.0, -1.45), (0.0, -1.72), (0.0, -1.95)]
_MIDDLE_CURLED = [(0.0, -1.0), (0.02, -1.3), (0.05, -1.0), (0.03, -0.8)]
_RING_UP = [(0.22, -0.95), (0.25, -1.35), (0.27, -1.6), (0.28, -1.8)]
_RING_CURLED = [(0.22, -0.95), (0.25, -1.2), (0.26, -0.95), (0.24, -0.78)]
_PINKY_UP = [(0.42, -0.85), (0.48, -1.15), (0.52, -1.35), (0.55, -1.5)]
_PINKY_CURLED = [(0.42, -0.85), (0.46, -1.05), (0.46, -0.85), (0.44, -0.72)]
_INDEX_V = [(-0.25, -0.95), (-0.33, -1.33), (-0.39, -1.58), (-0.45, -1.82)]  # index and middle spread in a V
_MIDDLE_V = [(0.0, -1.0), (0.06, -1.44), (0.1, -1.7), (0.14, -1.93)]
W0 = [(0.0, 0.0)]

SHAPES = {
    PEACE: W0 + _THUMB_TUCKED + _INDEX_V + _MIDDLE_V + _RING_CURLED + _PINKY_CURLED,
    L_SHAPE: W0 + _THUMB_OUT + _INDEX_UP + _MIDDLE_CURLED + _RING_CURLED + _PINKY_CURLED,
}
ONE = W0 + _THUMB_TUCKED + _INDEX_UP + _MIDDLE_CURLED + _RING_CURLED + _PINKY_CURLED  # one finger up
OPEN = W0 + _THUMB_OUT + _INDEX_UP + _MIDDLE_UP + _RING_UP + _PINKY_UP
LABEL = {PEACE: "Victory", L_SHAPE: "None"}  # MediaPipe has no class for the L


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


def shown(name: str, wrist=None, score: float = 0.7) -> Hand:
    return sign_hand(name, BESIDE if wrist is None else wrist, LABEL[name], score if name == PEACE else 0.9)


def run(det: PoseDetector, hands_per_frame, face=None, t0: float = 0.0) -> list[dict[str, bool]]:
    out = []
    for k, hands in enumerate(hands_per_frame):
        t = t0 + k * DT
        out.append(det.update(Signals(t, k, list(hands), face, width=W, height=H), t))
    return out


BESIDE = px(1.9, 1.2)  # wrist beside the face, below the cheek: the sign is held up next to the head


# ---------------------------------------------------------------- shapes
@pytest.mark.parametrize("name", HAND_SIGNS)
def test_each_shape_is_recognised_and_only_its_own(name):
    checks = {PEACE: peace_shape, L_SHAPE: l_shape}
    h = sign_hand(name, BESIDE)
    assert {n for n, f in checks.items() if f(h)} == {name}
    assert not thumb_up_shape(h)


def test_one_finger_open_hand_and_sideways_l_are_no_sign():
    for pts in (ONE, OPEN, [(y, x) for x, y in SHAPES[L_SHAPE]]):  # the last one points at image-left
        h = sign_hand(pts, BESIDE)
        assert not (peace_shape(h) or l_shape(h))


def test_snap_loaded_hand_is_not_an_l():
    pts = list(SHAPES[L_SHAPE])
    pts[4] = pts[12]  # thumb tip pressed to the middle fingertip ...
    pts[9:13] = _MIDDLE_UP  # ... of a straight middle finger
    assert not l_shape(sign_hand(pts, BESIDE))


# ---------------------------------------------------------------- detector
@pytest.mark.parametrize("name", HAND_SIGNS)
def test_sign_detected_beside_the_face(name):
    res = run(PoseDetector(), [[shown(name)]] * 10, make_face())
    assert all(r[name] for r in res)
    assert not any(r[other] for r in res for other in HAND_SIGNS if other != name)


@pytest.mark.parametrize("name", HAND_SIGNS)
def test_sign_detected_without_a_face(name):
    assert all(r[name] for r in run(PoseDetector(), [[shown(name)]] * 5))


def test_peace_label_needs_the_shape_unless_sure():
    assert not any(r[PEACE] for r in run(PoseDetector(), [[sign_hand(OPEN, BESIDE, "Victory", 0.7)]] * 5))
    assert all(r[PEACE] for r in run(PoseDetector(), [[sign_hand(OPEN, BESIDE, "Victory", 0.85)]] * 5))


def test_peace_by_shape_alone_needs_the_strict_shape():
    # MediaPipe often says "None" for a peace sign shown with the back of the hand
    for label, score in (("None", 0.9), ("Victory", 0.4)):
        assert all(r[PEACE] for r in run(PoseDetector(), [[sign_hand(PEACE, BESIDE, label, score)]] * 5))
    assert not any(r[PEACE] for r in run(PoseDetector(), [[sign_hand(PEACE, BESIDE, "Open_Palm", 0.9)]] * 5))
    together = W0 + _THUMB_TUCKED + _INDEX_UP + _MIDDLE_UP + _RING_CURLED + _PINKY_CURLED  # no V
    three = W0 + _THUMB_OUT + _INDEX_V + _MIDDLE_V + _RING_CURLED + _PINKY_CURLED  # thumb out too
    snap = list(SHAPES[PEACE])
    snap[4] = snap[12]  # thumb pressed to the middle fingertip: a hand about to snap
    for pts in (together, three, snap):
        h = sign_hand(pts, BESIDE, "None", 0.9)
        assert peace_shape(h) and not peace_shape_strict(h)
        assert not any(r[PEACE] for r in run(PoseDetector(), [[h]] * 5))


def test_l_counts_while_the_classifier_sees_no_other_sign():
    assert all(r[L_SHAPE] for r in run(PoseDetector(), [[sign_hand(L_SHAPE, BESIDE, "None", 0.9)]] * 5))
    for label in ("Pointing_Up", "Victory", "Open_Palm"):
        assert not any(r[L_SHAPE] for r in run(PoseDetector(), [[sign_hand(L_SHAPE, BESIDE, label, 0.7)]] * 5))


def test_finger_on_the_face_is_not_a_sign():
    # an L with the index up the cheek (chin on hand); a peace sign over an eye
    cases = ((L_SHAPE, px(0.25, 1.35)), (PEACE, px(0.15, 0.9)))
    for name, wrist in cases:
        assert not any(r[name] for r in run(PoseDetector(), [[shown(name, wrist)]] * 8, make_face()))


def test_sign_held_right_beside_the_cheek_counts():
    pts = [(-x, y) for x, y in SHAPES[PEACE]]  # the other hand
    h = sign_hand(pts, (765.0, 520.0), "Victory", 0.8)  # its box overlaps the face's by ~30 %
    assert all(r[PEACE] for r in run(PoseDetector(), [[h]] * 8, make_face()))


def test_moving_hand_is_not_a_sign():
    frames = [[sign_hand(L_SHAPE, BESIDE + (k * 0.07 * FW, 0.0), "None", 0.9)] for k in range(15)]
    res = run(PoseDetector(), frames, make_face())  # ~2 face widths per second
    assert not any(r[L_SHAPE] for r in res[8:])


def test_hud_says_why_a_sign_does_not_count():
    det = PoseDetector()
    run(det, [[shown(L_SHAPE, px(0.25, 1.35))]] * 3, make_face())
    assert det.debug["sign_why"] == {L_SHAPE: "on your face"}
    run(det, [[sign_hand(OPEN, BESIDE, "Victory", 0.6)]] * 3, make_face(), t0=1.0)
    assert det.debug["sign_why"] == {PEACE: "fingers?"}
    frames = [[sign_hand(PEACE, BESIDE + (k * 0.07 * FW, 0.0), "Victory", 0.9)] for k in range(10)]
    run(det, frames, make_face(), t0=2.0)
    assert det.debug["sign_why"] == {PEACE: "hold still"}
    run(det, [[shown(PEACE)]] * 10, make_face(), t0=3.0)
    assert det.debug["sign_why"] == {}


# ---------------------------------------------------------------- arbiter
def _cfg():
    cfg = zap_config.defaults()
    cfg.app.armed_on_start = True
    cfg.app.arm_grace_s = 0.0
    cfg.zoom.leave_guard_s = 0.0
    return cfg


def _hold(name, seconds, busy=False, flicker=False):
    clock = ReplayClock(0.0)
    arb, det = Arbiter(_cfg(), clock), PoseDetector()
    out = []
    for k in range(round(seconds / DT)):
        t = k * DT
        clock.set(t)
        h = shown(name)
        if flicker and k % 9 == 4:  # one frame in nine the classifier loses it
            h.gesture, h.gesture_score = "None", 0.0
        truths = det.update(Signals(t, k, [h], make_face(), width=W, height=H), t)
        out += [(round(i.t, 2), i.gesture) for i in arb.update(truths, (), t, busy=busy, t_on=det.t_on)
                if i.kind == "trigger"]
    return out


@pytest.mark.parametrize("name", HAND_SIGNS)
def test_held_sign_fires_once(name):
    fired = _hold(name, 3.0)
    assert [g for _, g in fired] == [name]
    assert 0.7 <= fired[0][0] <= 1.0  # about a second of holding, no more


def test_flicker_to_none_is_tolerated():
    assert [g for _, g in _hold(PEACE, 3.0, flicker=True)] == [PEACE]


def test_quick_sign_does_not_fire():
    assert _hold(PEACE, 0.5) == [] and _hold(L_SHAPE, 0.5) == []


def test_sign_waits_while_a_meme_is_showing():
    assert _hold(L_SHAPE, 2.0, busy=True) == []


# ---------------------------------------------------------------- bindings and HUD
def test_default_bindings_one_gesture_per_effect():
    b = zap_config.defaults().bindings
    assert b[PEACE] == "brb" and b[L_SHAPE] == "tbc"
    assert b["snap2"] == "leave" and b["snap1"] == "presence"
    assert b["forehead"] == "cutaway:michael_scott" and b["eyebrows_up"] == "cutaway:jim"
    targets = [t for t in b.values() if t != "none"]
    assert len(targets) == len(set(targets)), "two gestures for one effect"
    assert not {"point_up", "thumbs_down", "rock"} & set(b)


def test_hud_rows_follow_the_bindings():
    rows = dict(pose_rows(zap_config.defaults().bindings))
    assert rows[PEACE] == "peace > right back"
    assert rows[L_SHAPE] == "L sign > to be cont."
    assert rows["forehead"] == "forehead > michael" and rows["eyebrows_up"] == "brows up > jim"


def test_card_config_sections(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[brb]\ntext = "BRB"\nseconds = 3\n[tbc]\nseconds = 5\n', encoding="utf-8")
    cfg = zap_config.load(p)
    assert cfg.brb.text == "BRB" and cfg.brb.seconds == 3.0 and cfg.brb.hold_when_away
    assert cfg.tbc.seconds == 5.0
    p.write_text("[brb]\nhold_when_awy = true\n", encoding="utf-8")
    with pytest.raises(zap_config.ConfigError, match="did you mean 'hold_when_away'"):
        zap_config.load(p)
    p.write_text('[bindings]\nsnap2 = "outro"\n', encoding="utf-8")  # renamed: leave / tbc
    with pytest.raises(zap_config.ConfigError, match="not an effect"):
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
