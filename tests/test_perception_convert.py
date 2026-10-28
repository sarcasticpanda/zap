"""Conversion of MediaPipe results into Signals types, using SimpleNamespace fakes (no models)."""

from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from zap.signals import FACE_NOSE_BRIDGE, MIDDLE_MCP, WRIST, Face
from zap.vision.perception import (
    FaceCache,
    MaskPolarity,
    Timings,
    convert_face,
    convert_hands,
    convert_mask,
)

W, H = 1280, 720


def lms(n: int, x: float = 0.25, y: float = 0.5, z: float = -0.1) -> list[NS]:
    return [NS(x=x + i * 1e-3, y=y, z=z) for i in range(n)]


def cat(name: str, score: float) -> NS:
    return NS(category_name=name, score=score, index=-1, display_name="")


def hand_result(n_hands: int = 2) -> NS:
    return NS(
        hand_landmarks=[lms(21, x=0.2 + 0.5 * k) for k in range(n_hands)],
        handedness=[[cat("Left", 0.9 - 0.1 * k)] for k in range(n_hands)],
        gestures=[[cat("Open_Palm", 0.8), cat("None", 0.1)] if k == 0 else [cat("None", 0.7)] for k in range(n_hands)],
        hand_world_landmarks=[],
    )


# ---------------------------------------------------------------- hands
def test_convert_hands_scales_to_output_pixels():
    hands = convert_hands(hand_result(2), W, H)
    assert len(hands) == 2
    h0 = hands[0]
    assert h0.lm.shape == (21, 3) and h0.lm.dtype == np.float32
    assert h0.lm[WRIST, 0] == pytest.approx(0.2 * W)
    assert h0.lm[WRIST, 1] == pytest.approx(0.5 * H)
    assert h0.lm[WRIST, 2] == pytest.approx(-0.1 * W)  # z scales like x
    assert h0.lm[MIDDLE_MCP, 0] == pytest.approx((0.2 + 9e-3) * W, rel=1e-5)
    assert hands[1].lm[WRIST, 0] == pytest.approx(0.7 * W)


def test_convert_hands_gesture_and_scores():
    hands = convert_hands(hand_result(2), W, H)
    assert (hands[0].gesture, hands[0].gesture_score) == ("Open_Palm", pytest.approx(0.8))
    assert hands[0].score == pytest.approx(0.9)
    assert (hands[1].gesture, hands[1].gesture_score) == ("None", pytest.approx(0.7))
    assert hands[1].score == pytest.approx(0.8)


def test_convert_hands_picks_highest_gesture_even_if_unsorted():
    r = hand_result(1)
    r.gestures = [[cat("None", 0.2), cat("Thumb_Up", 0.75)]]
    (h,) = convert_hands(r, W, H)
    assert h.gesture == "Thumb_Up" and h.gesture_score == pytest.approx(0.75)


def test_convert_hands_missing_gestures_and_handedness():
    r = NS(hand_landmarks=[lms(21)], handedness=[], gestures=[])
    (h,) = convert_hands(r, W, H)
    assert h.gesture == "None" and h.gesture_score == 0.0 and h.score == 0.0


def test_convert_hands_empty_and_none():
    assert convert_hands(None, W, H) == []
    assert convert_hands(NS(hand_landmarks=[], handedness=[], gestures=[]), W, H) == []


def test_convert_hands_skips_malformed():
    r = NS(hand_landmarks=[lms(5), lms(21)], handedness=[[cat("Left", 0.5)]] * 2, gestures=[[], []])
    assert len(convert_hands(r, W, H)) == 1


def test_convert_hands_size_property_works():
    (h,) = convert_hands(hand_result(1), W, H)
    assert h.size == pytest.approx(9e-3 * W, rel=1e-4)


# ---------------------------------------------------------------- face
def face_result(with_extras: bool = True) -> NS:
    pts = lms(478, x=0.3, y=0.4, z=0.02)
    return NS(
        face_landmarks=[pts],
        face_blendshapes=[[cat("browInnerUp", 0.6), cat("jawOpen", 0.1)]] if with_extras else [],
        facial_transformation_matrixes=[np.eye(4, dtype=np.float32) * 2] if with_extras else [],
    )


def test_convert_face_landmarks_blend_matrix():
    f = convert_face(face_result(), W, H, t=12.5)
    assert isinstance(f, Face)
    assert f.lm.shape == (478, 3) and f.lm.dtype == np.float32
    assert f.lm[0, 0] == pytest.approx(0.3 * W)
    assert f.lm[0, 1] == pytest.approx(0.4 * H)
    assert f.lm[0, 2] == pytest.approx(0.02 * W)
    assert f.blend == {"browInnerUp": pytest.approx(0.6), "jawOpen": pytest.approx(0.1)}
    assert f.matrix.shape == (4, 4) and f.matrix[0, 0] == 2
    assert f.t == 12.5 and f.fresh
    assert f.center[0] == pytest.approx(f.lm[FACE_NOSE_BRIDGE, 0])


def test_convert_face_flat_matrix_and_no_extras():
    r = face_result(with_extras=False)
    f = convert_face(r, W, H)
    assert f.blend == {} and f.matrix is None
    r.facial_transformation_matrixes = [list(range(16))]
    assert convert_face(r, W, H).matrix[1, 0] == 4


def test_convert_face_none_when_nothing_found():
    assert convert_face(None, W, H) is None
    assert convert_face(NS(face_landmarks=[], face_blendshapes=[], facial_transformation_matrixes=[]), W, H) is None


def test_convert_face_tolerates_none_coordinates():
    r = face_result()
    r.face_landmarks[0][3] = NS(x=None, y=None, z=None)
    f = convert_face(r, W, H)
    assert np.all(f.lm[3] == 0)


# ---------------------------------------------------------------- mask
class FakeImage:
    def __init__(self, a: np.ndarray) -> None:
        self.a = a
        self.a.flags.writeable = False

    def numpy_view(self) -> np.ndarray:
        return self.a


def test_convert_mask_squeezes_and_copies():
    src = np.full((360, 640, 1), 0.25, np.float32)
    m = convert_mask(NS(confidence_masks=[FakeImage(src)], category_mask=None))
    assert m.shape == (360, 640) and m.dtype == np.float32
    assert m.flags.writeable and not np.shares_memory(m, src)


def test_convert_mask_none_and_bad_shape():
    assert convert_mask(None) is None
    assert convert_mask(NS(confidence_masks=None)) is None
    assert convert_mask(NS(confidence_masks=[])) is None
    with pytest.raises(ValueError):
        convert_mask(NS(confidence_masks=[FakeImage(np.zeros((4, 4, 2), np.float32))]))


# ---------------------------------------------------------------- helpers
def make_face(t: float) -> Face:
    return Face(lm=np.zeros((478, 3), np.float32), t=t)


def test_face_cache_reuses_for_one_second_then_drops():
    c = FaceCache(max_age=1.0)
    f = make_face(10.0)
    assert c.update(f, 10.0) is f
    stale = c.update(None, 10.5)
    assert stale is not None and stale.fresh is False and stale.t == 10.0
    assert c.update(None, 10.9) is stale  # same stale object, not rebuilt every frame
    assert c.update(None, 11.01) is None
    assert c.update(None, 10.5) is None  # dropped for good


def test_face_cache_new_face_replaces_stale():
    c = FaceCache()
    c.update(make_face(1.0), 1.0)
    c.update(None, 1.5)
    g = make_face(1.6)
    assert c.update(g, 1.6) is g and g.fresh


def test_mask_polarity_person_confidence():
    m = np.zeros((360, 640), np.float32)
    m[100:300, 250:400] = 0.97
    pol = MaskPolarity()
    pol.observe(m, np.array([640.0, 400.0]), W, H)  # -> (320, 200) at inference res
    assert pol.person is True and pol.label == "person"
    out = pol.apply(m.copy())
    assert out[200, 320] == pytest.approx(0.97)


def test_mask_polarity_background_confidence_inverts():
    m = np.ones((360, 640), np.float32)
    m[100:300, 250:400] = 0.03
    pol = MaskPolarity()
    pol.observe(m, np.array([640.0, 400.0]), W, H)
    assert pol.person is False and pol.label == "background"
    out = pol.apply(m)
    assert out[200, 320] == pytest.approx(0.97) and out[0, 0] == pytest.approx(0.0)
    # decided once: later contradicting samples are ignored
    pol.observe(np.ones((360, 640), np.float32), np.array([640.0, 400.0]), W, H)
    assert pol.person is False


def test_mask_polarity_waits_on_ambiguous_then_uses_mean():
    pol = MaskPolarity(max_samples=3)
    for v in (0.55, 0.6):
        pol.observe(np.full((36, 64), v, np.float32), np.array([640.0, 360.0]), W, H)
        assert pol.person is None
        assert pol.apply(np.full((2, 2), 0.3, np.float32))[0, 0] == pytest.approx(0.3)  # passthrough
    pol.observe(np.full((36, 64), 0.58, np.float32), np.array([640.0, 360.0]), W, H)
    assert pol.person is True


def test_mask_polarity_ignores_face_outside_frame():
    pol = MaskPolarity()
    pol.observe(np.ones((36, 64), np.float32), np.array([-50.0, 100.0]), W, H)
    assert pol.person is None


def test_timings_last_and_p95():
    t = Timings(window=100)
    for i in range(1, 101):
        t.add("hands", float(i))
    assert t.last["hands"] == 100.0
    assert t.p95("hands") == pytest.approx(95.05)
    assert t.p95("face") == 0.0
