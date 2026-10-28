"""Face morph mesh warp, face-swap geometry and the two-thumbs-up pose."""

import numpy as np
import pytest

from zap.effects.faceswap import ARC_DST, five_points, is_frontal, umeyama
from zap.effects.morph import FACE_OVAL, INNER, Morph, control_points
from zap.gestures.poses import PoseDetector
from zap.gestures.types import TWO_THUMBS
from zap.signals import Hand, Signals

W, H = 320, 180


def _face(cx=160.0, cy=90.0, r=40.0) -> np.ndarray:
    rng = np.random.default_rng(1)
    lm = np.zeros((478, 3), np.float32)
    lm[:, :2] = (cx, cy)
    ang = np.linspace(0, 2 * np.pi, len(FACE_OVAL), endpoint=False)
    lm[list(FACE_OVAL), 0] = cx + r * np.cos(ang)
    lm[list(FACE_OVAL), 1] = cy + 1.2 * r * np.sin(ang)
    lm[list(INNER), :2] = np.stack([cx + rng.uniform(-0.5, 0.5, len(INNER)) * r,
                                    cy + rng.uniform(-0.7, 0.7, len(INNER)) * r], axis=1)
    return lm


def _picture(seed: int) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 4)


def test_morph_endpoints_are_the_two_pictures():
    a, b = _picture(1), _picture(2)
    pa, pb = control_points(_face()), control_points(_face(150, 95, 35))
    m, out = Morph(W, H), np.empty_like(a)
    m.render(a, pa, b, pb, 0.0, out)
    assert np.abs(out.astype(int) - a).mean() < 1.0
    m.render(a, pa, b, pb, 1.0, out)
    assert np.abs(out.astype(int) - b).mean() < 1.0


def test_morph_moves_the_face_between_the_two_positions():
    a = np.zeros((H, W, 3), np.uint8)
    lm = _face(100, 90)
    a[80:100, 90:110] = 255  # a bright patch on face A's centre
    b = np.zeros_like(a)
    pa, pb = control_points(lm), control_points(_face(220, 90))
    out = np.empty_like(a)
    Morph(W, H).render(a, pa, b, pb, 0.5, out, alpha=0.0)  # shapes halfway, colours all A
    _, xs = np.nonzero(out[..., 0] > 128)
    assert 150 < xs.mean() < 170  # the patch travelled with the face to the midpoint


def test_umeyama_recovers_a_similarity():
    src = ARC_DST.copy()
    th, s = 0.3, 1.7
    R = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    dst = src @ R.T + (5, -3)
    M = umeyama(src, dst)
    assert np.allclose(src @ M[:, :2].T + M[:, 2], dst, atol=1e-3)


def test_five_points_are_in_image_order_even_when_mirrored():
    lm = np.zeros((478, 3), np.float32)
    lm[[33, 133], 0], lm[[362, 263], 0] = 200, 100  # a mirrored face: "right" eye on the left
    lm[61, 0], lm[291, 0] = 180, 120
    k = five_points(lm)
    assert k[0, 0] < k[1, 0] and k[3, 0] < k[4, 0]


def test_is_frontal():
    lm = np.zeros((478, 3), np.float32)
    lm[[33, 133], :2] = (100, 100)
    lm[[362, 263], :2] = (160, 100)
    lm[61, :2], lm[291, :2] = (110, 160), (150, 160)
    lm[1, :2] = (130, 130)
    assert is_frontal(lm)
    lm[1, :2] = (155, 130)  # nose far toward one eye: head turned
    assert not is_frontal(lm)


@pytest.mark.parametrize(("gestures", "expect"), [
    (("Thumb_Up", "Thumb_Up"), True),
    (("Thumb_Up", "None"), False),
    (("Thumb_Up",), False),
])
def test_two_thumbs_up(gestures, expect):
    det = PoseDetector()
    hands = []
    for i, g in enumerate(gestures):
        lm = np.zeros((21, 3), np.float32)
        lm[:, 0] = 300 + 500 * i + np.linspace(0, 150, 21)
        lm[:, 1] = 200 + np.linspace(0, 300, 21)
        hands.append(Hand(lm, 0.95, g, 0.9))
    truths = det.update(Signals(t=1.0, seq=1, hands=hands, width=1280, height=720), 1.0)
    assert truths[TWO_THUMBS] is expect
