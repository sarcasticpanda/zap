from __future__ import annotations

import math

import numpy as np
import pytest

from zap.gestures.poses import PoseDetector, load_params, snap_loaded_hands
from zap.gestures.types import CANCEL, COLLAR, COVER_FACE, EYEBROWS_UP, FOREHEAD, HANDS_UP, THUMBS_UP
from zap.signals import Face, Hand, Signals
from zap.vision.geometry import face_frame

W, H = 1280, 720
DT = 1 / 30
FW, CX, CY = 200.0, 640.0, 300.0  # face width and nose bridge (168) in pixels

# Key face landmarks in face-frame units (u right, v down, face widths from the nose bridge).
KEY_UV = {
    10: (0.0, -0.5), 151: (0.0, -0.33), 9: (0.0, -0.15), 168: (0.0, 0.0), 1: (0.0, 0.33), 152: (0.0, 0.78),
    234: (-0.5, 0.05), 454: (0.5, 0.05), 105: (-0.25, -0.12), 334: (0.25, -0.12), 145: (-0.2, 0.08), 374: (0.2, 0.08),
}

# Open right-ish hand, fingers along +y, x across the palm; units of hand size s.
HAND_LOCAL = np.array([
    (0.0, 0.0),
    (-0.3, 0.2), (-0.5, 0.4), (-0.65, 0.6), (-0.8, 0.75),  # thumb
    (-0.25, 0.95), (-0.28, 1.35), (-0.3, 1.6), (-0.32, 1.8),  # index
    (0.0, 1.0), (0.0, 1.45), (0.0, 1.72), (0.0, 1.95),  # middle
    (0.22, 0.95), (0.25, 1.35), (0.27, 1.6), (0.28, 1.8),  # ring
    (0.42, 0.85), (0.48, 1.15), (0.52, 1.35), (0.55, 1.5),  # pinky
])
PALM_LOCAL = HAND_LOCAL[[0, 5, 9, 13, 17]].mean(axis=0)


def px(u: float, v: float, cx: float = CX, cy: float = CY, fw: float = FW) -> np.ndarray:
    return np.array([cx + u * fw, cy + v * fw])


def make_face(blend=None, fresh=True, yaw_deg=None, cx=CX, cy=CY, fw=FW) -> Face:
    n = 478
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.sqrt(np.linspace(0.05, 1.0, n))
    u, v = 0.48 * r * np.cos(ang), 0.14 + 0.62 * r * np.sin(ang)
    for i, (uu, vv) in KEY_UV.items():
        u[i], v[i] = uu, vv
    lm = np.zeros((n, 3), np.float32)
    lm[:, 0], lm[:, 1] = cx + u * fw, cy + v * fw
    matrix = None
    if yaw_deg is not None:
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        matrix = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], float)
    return Face(lm, dict(blend or {}), matrix, 0.0, fresh)


def make_hand(palm, s=0.7 * FW, angle=0.0, score=0.95, gesture="None", gscore=0.0, curl=False, mirror=False) -> Hand:
    """angle 0 = fingers up, -90 = fingers toward image-left, 90 = toward image-right.

    The thumb points toward image-left; ``mirror`` makes the other hand (thumb toward image-right).
    """
    loc = HAND_LOCAL.copy()
    if curl:
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18), (7, 6), (11, 10), (15, 14), (19, 18)):
            loc[tip] = (loc[pip][0], 0.6)
    a = math.radians(angle)
    up, side = np.array([math.sin(a), -math.cos(a)]), np.array([math.cos(a), math.sin(a)])
    rel = loc - PALM_LOCAL
    if mirror:
        rel[:, 0] = -rel[:, 0]
    lm = np.zeros((21, 3), np.float32)
    lm[:, :2] = np.asarray(palm, float) + rel[:, :1] * side * s + rel[:, 1:2] * up * s
    return Hand(lm, score, gesture, gscore)


def run(det: PoseDetector, frames, t0: float = 0.0) -> list[dict[str, bool]]:
    """frames: list of (hands, face); one frame every DT."""
    out = []
    for k, (hands, face) in enumerate(frames):
        t = t0 + k * DT
        out.append(det.update(Signals(t, k, list(hands), face, width=W, height=H), t))
    return out


def held(hands, face, n=10):
    return [(hands, face)] * n


def hands_up_pair(u=1.2, v=-0.2, **face_kw):
    return [make_hand(px(-u, v, **face_kw), mirror=True), make_hand(px(u, v, **face_kw))]


# ---------------------------------------------------------------- face fixture sanity
def test_synthetic_face_frame():
    ff = face_frame(make_face())
    assert ff.fw == pytest.approx(FW)
    assert ff.to_uv(px(0.3, -0.4)[None])[0] == pytest.approx([0.3, -0.4])


# ---------------------------------------------------------------- hands_up
def test_hands_up_positive():
    res = run(PoseDetector(), held(hands_up_pair(), make_face()))
    assert all(r[HANDS_UP] for r in res)
    assert not any(r[FOREHEAD] or r[COLLAR] or r[COVER_FACE] for r in res)


def test_hands_up_with_cached_face():
    res = run(PoseDetector(), held(hands_up_pair(), make_face(fresh=False)))
    assert all(r[HANDS_UP] for r in res)


def test_hands_up_overhead_stretch_rejected():
    kw = {"cy": 450.0, "fw": 150.0}
    face = make_face(**kw)
    hands = [make_hand(px(-1.0, -2.0, **kw), s=105), make_hand(px(1.0, -2.0, **kw), s=105)]
    assert not any(r[HANDS_UP] for r in run(PoseDetector(), held(hands, face)))
    hands = [make_hand(px(-1.0, -0.5, **kw), s=105), make_hand(px(1.0, -0.5, **kw), s=105)]
    assert all(r[HANDS_UP] for r in run(PoseDetector(), held(hands, face)))


@pytest.mark.parametrize("case", ["one_hand", "fists", "too_close", "same_side", "no_face", "low_score"])
def test_hands_up_negatives(case):
    face, hands = make_face(), hands_up_pair()
    if case == "one_hand":
        hands = hands[:1]
    elif case == "fists":
        hands = [make_hand(h.palm_center, curl=True) for h in hands]
    elif case == "too_close":
        hands = hands_up_pair(u=0.65)  # |u| ok but gap 1.3 fw
    elif case == "same_side":
        hands = [make_hand(px(0.8, -0.2)), make_hand(px(2.2, -0.2))]
    elif case == "no_face":
        face = None
    elif case == "low_score":
        hands[0].score = 0.4
    assert not any(r[HANDS_UP] for r in run(PoseDetector(), held(hands, face)))


def test_talking_hands_blank_static_poses():
    det = PoseDetector()
    frames = []
    for k in range(10):
        dx = 40.0 * (k % 2)  # 40 px per frame = 6 fw/s
        hands = [make_hand(px(-1.2, -0.2) + (dx, 0)), make_hand(px(1.2, -0.2) - (dx, 0))]
        frames.append((hands, make_face()))
    res = run(det, frames)
    assert not any(r[HANDS_UP] for r in res[1:])
    assert det.debug["talking"]


# ---------------------------------------------------------------- forehead
def forehead_hand(u=0.35, v=-0.4, **kw):
    return make_hand(px(u, v), angle=-90, **kw)  # fingers across the forehead toward image-left


def test_forehead_positive_sideways_hand():
    det = PoseDetector()
    res = run(det, held([forehead_hand()], make_face()))
    assert all(r[FOREHEAD] for r in res)
    assert det.debug[FOREHEAD]["tips"] >= 2
    assert not any(r[HANDS_UP] or r[COLLAR] for r in res)


def test_forehead_positive_upright_palm_on_forehead():
    res = run(PoseDetector(), held([make_hand(px(0.0, -0.1))], make_face()))
    assert all(r[FOREHEAD] for r in res)


def test_facepalm_over_eyes_is_not_forehead():
    det = PoseDetector()
    res = run(det, held([make_hand(px(0.0, 0.15))], make_face()))  # fingertips on forehead, palm on eyes
    assert det.debug[FOREHEAD]["tips"] >= 2
    assert not any(r[FOREHEAD] for r in res)


def test_scratching_forehead_rejected():
    frames = []
    for k in range(15):
        h = forehead_hand()
        h.lm[[8, 12, 16, 20], 1] += (10.0 if k % 2 else -10.0)  # fingertips wiggle, palm still
        frames.append(([h], make_face()))
    res = run(PoseDetector(), frames)
    assert not any(r[FOREHEAD] for r in res[4:])


def test_forehead_moving_hand_rejected():
    frames = [([forehead_hand(u=0.35 - 0.03 * k)], make_face()) for k in range(10)]  # 0.9 fw/s
    res = run(PoseDetector(), frames)
    assert not any(r[FOREHEAD] for r in res[1:])


def test_forehead_tiny_hand_rejected():
    res = run(PoseDetector(), held([forehead_hand(s=0.3 * FW)], make_face()))
    assert not any(r[FOREHEAD] for r in res)


def test_forehead_single_frame_jitter_tolerated():
    # one landmark frame 15 px off (1.6 fw/s frame to frame) must not break a resting hand
    frames = [([forehead_hand(u=0.35 + (0.075 if k == 6 else 0.0))], make_face()) for k in range(12)]
    res = run(PoseDetector(), frames)
    assert all(r[FOREHEAD] for r in res)


# ---------------------------------------------------------------- forehead occlusion (hand lost on the forehead)
def lost_on_forehead(n_seen=10, n_lost=60, face=None, hand=None):
    face = face or make_face()
    return held([hand or forehead_hand()], face, n_seen) + held([], face, n_lost)


def test_forehead_hand_lost_by_the_model_still_counts():
    det = PoseDetector()
    res = run(det, lost_on_forehead())
    assert all(r[FOREHEAD] for r in res)  # seen, then lost while the face stays tracked
    assert det.occluded and det.debug["occluded"]


def test_forehead_occlusion_ends_when_the_hand_reappears_elsewhere():
    det = PoseDetector()
    frames = lost_on_forehead(n_lost=20) + held([make_hand(px(2.0, 1.2))], make_face(), 5)
    res = run(det, frames)
    assert all(r[FOREHEAD] for r in res[:30])
    assert not any(r[FOREHEAD] for r in res[30:]) and not det.occluded


def test_forehead_occlusion_times_out():
    res = run(PoseDetector(), lost_on_forehead(n_lost=200))  # 6.7 s
    assert res[10 + 170][FOREHEAD]  # 5.7 s after vanishing
    assert not res[-1][FOREHEAD]


def test_forehead_occlusion_needs_the_face():
    frames = held([forehead_hand()], make_face(), 10) + held([], None, 30)
    res = run(PoseDetector(), frames)
    assert not any(r[FOREHEAD] for r in res[10 + 16:])  # face gone (> face_hold_s)


@pytest.mark.parametrize("case", ["moving", "edge", "two_hands", "brief"])
def test_forehead_occlusion_negatives(case):
    det = PoseDetector()
    if case == "moving":  # passing through the region at 1.8 fw/s, then lost
        frames = [([forehead_hand(u=-0.5 + 0.06 * k)], make_face()) for k in range(10)] + held([], make_face(), 30)
    elif case == "edge":  # face near the top of the frame: the hand left through the top edge
        kw = {"cy": 230.0}  # without the fingertip tweak this hand is well inside and does count
        face = make_face(**kw)
        hand = make_hand(px(0.0, -0.45, **kw), angle=-90)
        hand.lm[[8, 12], 1] = 20.0  # fingertips within 5% of the top border
        frames = held([hand], face, 10) + held([], face, 30)
    elif case == "two_hands":  # both hands vanish over the face together: that's cover_face
        frames = held([forehead_hand(u=-0.3), forehead_hand(u=0.35)], make_face(), 10) + held([], make_face(), 30)
    else:  # seen in the region for a single frame only
        frames = held([make_hand(px(2.0, 1.2))], make_face(), 5) + held([forehead_hand()], make_face(), 1) + \
            held([], make_face(), 30)
    res = run(det, frames)
    assert not any(r[FOREHEAD] for r in res[-25:])
    assert not det.occluded


def test_forehead_occlusion_blocks_eyebrows():
    det = PoseDetector()
    frames = lost_on_forehead(n_lost=40, face=make_face(brows(0.9)))
    assert not any(r[EYEBROWS_UP] for r in run(det, frames))


# ---------------------------------------------------------------- cover_face
def covering_pair():
    return [make_hand(px(-0.2, 0.15)), make_hand(px(0.2, 0.15))]


def test_cover_face_fires_once_when_hands_drop():
    det = PoseDetector()
    frames = held(covering_pair(), make_face(fresh=False), 30) + held([], make_face(), 10)
    res = run(det, frames)
    fired = [k for k, r in enumerate(res) if r[COVER_FACE]]
    assert fired == [30]


def test_cover_face_with_face_lost_while_covered():
    frames = held(covering_pair(), make_face(), 2) + held(covering_pair(), None, 28) + held([], make_face(), 5)
    fired = [k for k, r in enumerate(run(PoseDetector(), frames)) if r[COVER_FACE]]
    assert fired == [30]


def test_cover_face_too_short():
    frames = held(covering_pair(), make_face(fresh=False), 9) + held([], make_face(), 10)
    assert not any(r[COVER_FACE] for r in run(PoseDetector(), frames))


def test_cover_face_face_never_returns():
    frames = held(covering_pair(), None, 30) + held([], None, 60) + held([], make_face(), 5)
    # the face was never seen, so there is no face box: nothing counts as covering
    assert not any(r[COVER_FACE] for r in run(PoseDetector(), frames))
    frames = held(covering_pair(), make_face(fresh=False), 30) + held([], None, 60) + held([], make_face(), 5)
    assert not any(r[COVER_FACE] for r in run(PoseDetector(), frames))  # back after > 1.5 s


def test_cover_face_fires_as_the_hands_come_off():
    one = covering_pair()[:1]
    frames = held(covering_pair(), make_face(fresh=False), 30) + held(one, make_face(), 10) + held([], make_face(), 3)
    fired = [k for k, r in enumerate(run(PoseDetector(), frames)) if r[COVER_FACE]]
    assert fired == [30]  # one hand away and the face seen again is enough


@pytest.mark.parametrize(("margin", "fires"), [(0.1, False), (10.0, True)])
def test_hands_beside_head_are_not_covering(margin, fires):
    # Hand bboxes overlap the face bbox by > 40%, but the palms are beside the face, not on it.
    frames = held(hands_up_pair(u=0.62, v=0.1), make_face(), 30) + held([], make_face(), 10)
    det = PoseDetector({"cover_face": {"palm_margin": margin}})
    assert any(r[COVER_FACE] for r in run(det, frames)) == fires


# ---------------------------------------------------------------- eyebrows_up
RAISED = {"browInnerUp": 0.8, "browOuterUpLeft": 0.7, "browOuterUpRight": 0.7, "jawOpen": 0.05}


def test_eyebrows_positive():
    res = run(PoseDetector(), held([], make_face(RAISED)))
    assert all(r[EYEBROWS_UP] for r in res)


@pytest.mark.parametrize("case", ["asym", "yaw", "jaw", "hand_near", "low"])
def test_eyebrows_negatives(case):
    blend, kw, hands = dict(RAISED), {}, []
    if case == "asym":
        blend.update(browOuterUpLeft=0.95, browOuterUpRight=0.45)  # mean still 0.73: only the asymmetry rejects
    elif case == "yaw":
        kw["yaw_deg"] = 25.0
    elif case == "jaw":
        blend["jawOpen"] = 0.5
    elif case == "hand_near":
        hands = [make_hand(px(0.8, 0.0), curl=True)]
    elif case == "low":
        blend.update(browInnerUp=0.3)  # mean 0.57 < 0.7
    res = run(PoseDetector(), held(hands, make_face(blend, **kw)))
    assert not any(r[EYEBROWS_UP] for r in res)


def test_eyebrows_blocked_by_any_hand_in_view():
    # brows go up while concentrating on a hand gesture: a real user's snaps set Jim off
    res = run(PoseDetector(), held([make_hand(px(2.5, 1.2), curl=True)], make_face(RAISED)))
    assert not res[-1][EYEBROWS_UP]
    res = run(PoseDetector({"eyebrows_up": {"hand_near_fw": 1.6}}),
              held([make_hand(px(2.5, 1.2), curl=True)], make_face(RAISED)))
    assert res[-1][EYEBROWS_UP]  # the old rule (hands near the face only) is still a setting


def brows(level: float, **extra) -> dict[str, float]:
    return {"browInnerUp": level, "browOuterUpLeft": level, "browOuterUpRight": level, "jawOpen": 0.0, **extra}


def test_eyebrows_threshold_follows_the_users_baseline():
    det = PoseDetector()
    run(det, held([], make_face(brows(0.2)), 90))  # 3 s relaxed
    assert det.debug[EYEBROWS_UP]["base"] == pytest.approx(0.2)
    assert det.debug[EYEBROWS_UP]["threshold"] == pytest.approx(0.7)  # max(0.7, 0.2 + 0.4)
    assert not run(det, held([], make_face(brows(0.57)), 3), t0=3.0)[-1][EYEBROWS_UP]
    assert run(det, held([], make_face(brows(0.9)), 3), t0=3.1)[-1][EYEBROWS_UP]


def test_eyebrows_concentration_level_is_not_a_raise():
    # brows at 0.55 (measured while doing other gestures) after a relaxed minute never count
    det = PoseDetector()
    run(det, held([], make_face(brows(0.2)), 60))
    assert not any(r[EYEBROWS_UP] for r in run(det, held([], make_face(brows(0.55)), 60), t0=2.0))


def test_eyebrows_blocked_shortly_after_a_hand_leaves():
    det = PoseDetector()
    hand = [make_hand(px(0.8, 0.0), curl=True)]
    run(det, held(hand, make_face(brows(0.9)), 10))
    res = run(det, held([], make_face(brows(0.9)), 30), t0=10 * DT)
    first = next(k for k, r in enumerate(res) if r[EYEBROWS_UP])
    assert 0.4 <= first * DT + DT <= 0.5  # hand_clear_s = 0.4


def test_eyebrows_baseline_ignores_frames_with_hands():
    det = PoseDetector()
    run(det, held([make_hand(px(0.8, 0.0), curl=True)], make_face(brows(0.6)), 90))
    assert det.debug[EYEBROWS_UP]["base"] == pytest.approx(0.2)  # still the start value


def test_eyebrows_baseline_for_naturally_high_brows():
    det = PoseDetector()
    rest = {"browInnerUp": 0.6, "browOuterUpLeft": 0.5, "browOuterUpRight": 0.5, "jawOpen": 0.05}
    res = run(det, held([], make_face(rest), 150))  # 5 s of resting face
    assert not res[-1][EYEBROWS_UP]
    up = {"browInnerUp": 0.95, "browOuterUpLeft": 0.9, "browOuterUpRight": 0.9, "jawOpen": 0.05}
    res = run(det, held([], make_face(up), 15), t0=5.0)
    assert res[-1][EYEBROWS_UP]


def test_eyebrows_never_while_speaking():
    # a real user's brows rose as high while talking as when raising them on purpose
    det = PoseDetector()
    frames = [([], make_face({**RAISED, "jawOpen": 0.2 * (k % 2)})) for k in range(30)]
    res = run(det, frames)
    assert det.debug[EYEBROWS_UP]["speaking"] and not res[-1][EYEBROWS_UP]
    det2 = PoseDetector()
    assert run(det2, held([], make_face(RAISED), 30))[-1][EYEBROWS_UP]


def test_eyebrows_speaking_only_doubles_hold_when_allowed():
    det = PoseDetector({"eyebrows_up": {"quiet_when_speaking": False}})
    run(det, [([], make_face({**RAISED, "jawOpen": 0.2 * (k % 2)})) for k in range(30)])
    assert det.t_on[EYEBROWS_UP] == pytest.approx(0.9)


def test_eyebrows_expire_without_fresh_face():
    det = PoseDetector()
    run(det, held([], make_face(RAISED), 5))
    res = run(det, held([], None, 30), t0=5 * DT)
    assert not res[-1][EYEBROWS_UP]


def test_pained_face_speeds_up_forehead():
    det = PoseDetector()
    run(det, held([], make_face({"mouthFrownLeft": 0.3, "mouthFrownRight": 0.3}), 5))
    assert det.t_on[FOREHEAD] == pytest.approx(0.5)
    det2 = PoseDetector()
    run(det2, held([], make_face(), 5))
    assert det2.t_on[FOREHEAD] == pytest.approx(0.7)


# ---------------------------------------------------------------- collar
def test_collar_fingertips_positive():
    res = run(PoseDetector(), held([make_hand(px(0.4, 1.13), angle=-90)], make_face()))
    assert all(r[COLLAR] for r in res)
    assert not any(r[FOREHEAD] for r in res)


def collar_pinch_hand(pinch=True):
    h = make_hand(px(0.3, 1.25), s=0.5 * FW)
    h.lm[20, :2] = px(0.55, 0.8)  # pinky out of the collar band: only the index tip is inside
    h.lm[8, :2] = px(0.2, 1.0)
    h.lm[4, :2] = px(0.22, 1.0) if pinch else px(-0.3, 1.2)
    return h


def test_collar_pinch_positive():
    assert all(r[COLLAR] for r in run(PoseDetector(), held([collar_pinch_hand()], make_face())))


def test_collar_without_pinch_or_tips():
    assert not any(r[COLLAR] for r in run(PoseDetector(), held([collar_pinch_hand(False)], make_face())))


def test_collar_moving_hand_rejected():
    frames = [([make_hand(px(0.4 - 0.05 * k, 1.13), angle=-90)], make_face()) for k in range(10)]  # 1.5 fw/s
    assert not any(r[COLLAR] for r in run(PoseDetector(), frames)[1:])


# ---------------------------------------------------------------- cancel / thumbs up / snaps
def test_cancel_open_palm():
    assert run(PoseDetector(), [([make_hand(px(1.5, 0.5))], None)])[0][CANCEL]
    assert not run(PoseDetector(), [([make_hand(px(1.5, 0.5), curl=True)], None)])[0][CANCEL]
    gestured = make_hand(px(1.5, 0.5), curl=True, gesture="Open_Palm", gscore=0.8)
    assert run(PoseDetector(), [([gestured], None)])[0][CANCEL]


def test_thumbs_up():
    h = make_hand(px(1.5, 0.5), curl=True, gesture="Thumb_Up", gscore=0.8)
    assert run(PoseDetector(), [([h], None)])[0][THUMBS_UP]
    h.gesture_score = 0.6
    assert not run(PoseDetector(), [([h], None)])[0][THUMBS_UP]


def loaded_hand(palm):
    h = make_hand(palm)
    h.lm[4, :2] = h.lm[12, :2] + (4.0, 0.0)
    return h


def test_snap_loaded_hands():
    sig = Signals(0.0, 0, [loaded_hand(px(1.8, 0.3)), make_hand(px(-1.8, 0.3))], None)
    loaded = snap_loaded_hands(sig)
    assert len(loaded) == 1
    assert loaded[0] == pytest.approx(px(1.8, 0.3))


def test_detector_ignores_loaded_snap_on_face():
    det = PoseDetector()
    run(det, [([loaded_hand(px(1.8, 0.3))], make_face())])
    assert len(det.snap_loaded) == 1
    run(det, [([loaded_hand(px(0.0, -0.2))], make_face())], t0=1.0)
    assert det.snap_loaded == []


def test_collar_pinch_is_not_a_loaded_snap():
    # real recording: fingers on the collar beside the chin read as thumb-on-middle-finger
    det = PoseDetector()
    run(det, [([loaded_hand(px(0.8, 0.78))], make_face())])
    assert det.snap_loaded == []
    run(det, [([loaded_hand(px(-2.0, 0.4))], make_face())], t0=1.0)  # snapping beside the face still counts
    assert len(det.snap_loaded) == 1


# ---------------------------------------------------------------- usable hands
@pytest.mark.parametrize("case", ["low_wrist", "at_edge", "tiny"])
def test_unusable_hands_are_ignored(case):
    hands = hands_up_pair()
    if case == "low_wrist":
        hands[0].lm[0, 1] = 1.3 * H  # well below the picture (just below it still counts)
    elif case == "at_edge":
        hands[0].lm[4, 0] = 2.0
    elif case == "tiny":
        hands[0] = make_hand(hands[0].palm_center, s=40.0)
    assert not any(r[HANDS_UP] for r in run(PoseDetector(), held(hands, make_face())))


# ---------------------------------------------------------------- config
def test_overrides_apply():
    det = PoseDetector({"hands_up": {"min_gap": 3}})
    assert det.params[HANDS_UP].min_gap == 3.0
    assert not any(r[HANDS_UP] for r in run(det, held(hands_up_pair(), make_face())))


@pytest.mark.parametrize(
    ("over", "hint"),
    [
        ({"hands_up": {"t_onn": 1.0}}, "t_on"),
        ({"hand_up": {"t_on": 1.0}}, "hands_up"),
        ({"forehead": {"min_tips": 2.5}}, "int"),
        ({"arbiter": {"setle_s": 0.2}}, "settle_s"),
    ],
)
def test_bad_overrides_raise(over, hint):
    with pytest.raises(ValueError, match=hint):
        PoseDetector(over)


def test_load_params_defaults():
    p = load_params()
    assert (p[HANDS_UP].t_on, p[HANDS_UP].t_off, p[HANDS_UP].min_show, p[HANDS_UP].cooldown) == (0.4, 0.6, 2.5, 4.0)
    assert p[FOREHEAD].dwell().t_on == 0.7
    assert p[COVER_FACE].cooldown == 10.0


def test_eyebrows_calibrated_rise_follows_the_current_resting_level():
    # calibrated: brows rise 0.2 when raised; today they rest at 0.4 (they rested at 0.8 when calibrating)
    det = PoseDetector()
    det.set_brow_calibration({"signal": "blend", "rise": 0.2})
    run(det, held([], make_face(brows(0.4)), 90))  # 3 s at rest
    assert det.debug[EYEBROWS_UP]["threshold"] == pytest.approx(0.4 + 0.6 * 0.2)
    assert not run(det, held([], make_face(brows(0.48)), 30), t0=3.0)[-1][EYEBROWS_UP]
    assert run(det, held([], make_face(brows(0.6)), 30), t0=4.0)[-1][EYEBROWS_UP]


def test_eyebrows_holding_a_raise_does_not_lift_the_resting_level():
    det = PoseDetector()
    det.set_brow_calibration({"signal": "blend", "rise": 0.2})
    run(det, held([], make_face(brows(0.4)), 300))  # 10 s at rest
    res = run(det, held([], make_face(brows(0.6)), 90), t0=10.0)  # a raise held for 3 s
    assert all(r[EYEBROWS_UP] for r in res[5:]), "the raise stopped counting while held"
    assert det.debug[EYEBROWS_UP]["base"] == pytest.approx(0.4, abs=0.01)


def test_eyebrows_a_new_resting_level_is_learned_again():
    det = PoseDetector()
    det.set_brow_calibration({"signal": "blend", "rise": 0.2})
    run(det, held([], make_face(brows(0.3)), 300))
    res = run(det, held([], make_face(brows(0.6)), 900), t0=10.0)  # the light changed: 0.6 is now "resting"
    assert not res[-1][EYEBROWS_UP] and det.debug[EYEBROWS_UP]["base"] == pytest.approx(0.6, abs=0.02)
