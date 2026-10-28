"""Perception.run scheduling, caching and robustness with fake MediaPipe tasks (no models, no camera)."""

from __future__ import annotations

import itertools
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from zap.bench import camera_candidates, pick_physical
from zap.config import PerceptionCfg
from zap.signals import FramePacket
from zap.vision.perception import Perception, TaskSet

W, H = 1280, 720


def lms(n: int, x: float, y: float) -> list[NS]:
    return [NS(x=x, y=y, z=0.0) for _ in range(n)]


class FakeImage:
    def __init__(self, a: np.ndarray) -> None:
        self.a = a

    def numpy_view(self) -> np.ndarray:
        return self.a


class Fake:
    """Records calls; `face_found`/`raise_on` steer what each task returns."""

    def __init__(self, background_mask: bool = False) -> None:
        self.calls: dict[str, list[int]] = {"hands": [], "face": [], "mask": []}
        self.shapes: list[tuple[int, ...]] = []
        self.threads: set[str] = set()
        self.face_found = True
        self.raise_on: set[str] = set()
        self.background_mask = background_mask
        self.closed = 0

    def _hit(self, key: str, image: np.ndarray, ts: int) -> None:
        self.calls[key].append(ts)
        self.shapes.append(image.shape)
        self.threads.add(threading.current_thread().name)
        if key in self.raise_on:
            raise RuntimeError(f"boom {key}")

    def recognize_for_video(self, image, ts):
        self._hit("hands", image, ts)
        return NS(
            hand_landmarks=[lms(21, 0.25, 0.5)],
            handedness=[[NS(category_name="Right", score=0.95)]],
            gestures=[[NS(category_name="Victory", score=0.66)]],
        )

    def detect_for_video(self, image, ts):
        self._hit("face", image, ts)
        if not self.face_found:
            return NS(face_landmarks=[], face_blendshapes=[], facial_transformation_matrixes=[])
        return NS(
            face_landmarks=[lms(478, 0.5, 0.5)],
            face_blendshapes=[[NS(category_name="browInnerUp", score=0.3)]],
            facial_transformation_matrixes=[np.eye(4)],
        )

    def segment_for_video(self, image, ts):
        self._hit("mask", image, ts)
        m = np.zeros((image.shape[0], image.shape[1], 1), np.float32)
        h, w = image.shape[:2]
        m[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 0.95  # person in the middle
        if self.background_mask:
            m = 1.0 - m
        return NS(confidence_masks=[FakeImage(m)])

    def close(self):
        self.closed += 1


def make(fake: Fake, pooled: bool = True, warmup: bool = False, **cfg) -> Perception:
    tasks = TaskSet(hands=fake, face=fake, seg=fake, wrap=lambda rgb: rgb)
    return Perception(PerceptionCfg(**cfg), W, H, pooled=pooled, warmup=warmup, tasks=tasks)


def pkt(i: int, t: float | None = None) -> FramePacket:
    frame = np.zeros((H, W, 3), np.uint8)
    frame[..., 0] = 255  # blue in BGR
    return FramePacket(frame=frame, t=1.0 + i / 30 if t is None else t, seq=i)


@pytest.mark.parametrize("pooled", [True, False])
def test_run_produces_signals_in_output_pixels(pooled):
    fake = Fake()
    with make(fake, pooled=pooled) as p:
        sig = p.run(pkt(0))
    assert (sig.width, sig.height, sig.seq) == (W, H, 0)
    (hand,) = sig.hands
    assert hand.lm[0, 0] == pytest.approx(0.25 * W) and hand.gesture == "Victory" and hand.score == pytest.approx(0.95)
    assert sig.face is not None and sig.face.fresh and sig.face.lm[0, 1] == pytest.approx(0.5 * H)
    assert sig.face.t == pytest.approx(1.0)
    assert sig.mask.shape == (360, 640) and sig.mask_fresh
    assert fake.shapes[0] == (360, 640, 3)
    assert fake.closed == 3  # one fake stands in for all three tasks


def test_frame_is_rgb_at_inference_size():
    seen = []
    fake = Fake()
    tasks = TaskSet(hands=fake, face=fake, seg=fake, wrap=lambda rgb: seen.append(rgb[0, 0].copy()) or rgb)
    p = Perception(PerceptionCfg(infer_width=320, infer_height=180), W, H, warmup=False, tasks=tasks)
    p.run(pkt(0))
    p.close()
    assert list(seen[0]) == [0, 0, 255]  # BGR blue -> RGB
    assert fake.shapes[0] == (180, 320, 3)


def test_cadence_and_need_flags():
    fake = Fake()
    p = make(fake, face_every=2, mask_every=3)
    ran = []
    for i in range(7):
        p.run(pkt(i))
        ran.append(p.ran)
    assert all("hands" in r for r in ran)
    assert [("face" in r) for r in ran] == [True, False, True, False, True, False, True]
    assert [("mask" in r) for r in ran] == [True, False, False, True, False, False, True]
    p.run(pkt(7), need_face=True, need_mask=True)
    assert p.ran == ("hands", "face", "mask")
    p.close()


def test_timestamps_strictly_increasing_per_task_even_if_clock_repeats():
    fake = Fake()
    p = make(fake, face_every=1, mask_every=1, warmup=True)
    for i in range(5):
        p.run(pkt(i, t=0.0))  # e.g. a replay clock stuck at zero
    p.close()
    for key, ts in fake.calls.items():
        assert all(b > a for a, b in itertools.pairwise(ts)), key


def test_face_cache_and_stale_flag():
    fake = Fake()
    p = make(fake, face_every=1)
    assert p.run(pkt(0, t=5.0)).face.fresh
    fake.face_found = False
    s = p.run(pkt(1, t=5.5))
    assert s.face is not None and not s.face.fresh and s.face.t == 5.0
    assert p.run(pkt(2, t=6.2)).face is None
    p.close()


def test_mask_kept_between_refreshes():
    fake = Fake()
    p = make(fake, mask_every=3)
    s0 = p.run(pkt(0))
    s1 = p.run(pkt(1))
    assert s0.mask_fresh and not s1.mask_fresh and s1.mask is s0.mask
    p.close()


def test_background_mask_is_detected_and_inverted():
    fake = Fake(background_mask=True)
    p = make(fake)
    sig = p.run(pkt(0))
    assert p.mask_polarity == "background"
    assert sig.mask[180, 320] == pytest.approx(0.95, abs=1e-5)  # person at the centre after inversion
    assert sig.mask[0, 0] == pytest.approx(0.0, abs=1e-6)
    p.close()


def test_person_mask_left_alone():
    p = make(Fake())
    sig = p.run(pkt(0))
    assert p.mask_polarity == "person" and sig.mask[180, 320] == pytest.approx(0.95)
    p.close()


def test_failing_task_is_skipped_and_logged_once(caplog):
    fake = Fake()
    fake.raise_on = {"face"}
    p = make(fake, face_every=1)
    with caplog.at_level("WARNING"):
        sigs = [p.run(pkt(i)) for i in range(3)]
    assert all(len(s.hands) == 1 and s.mask is not None for s in sigs)
    assert all(s.face is None for s in sigs)
    assert p.errors["face"] == 3
    assert sum("face model failed" in r.getMessage() for r in caplog.records) == 1
    p.close()


def test_timings_reported():
    p = make(Fake())
    p.run(pkt(0))
    assert set(p.timings) >= {"hands", "face", "mask", "total"}
    assert p.timings["total"] >= p.timings["hands"] >= 0.0
    assert p.p95("total") > 0.0
    p.close()


def test_pooled_runs_on_worker_threads():
    fake = Fake()
    p = make(fake)
    p.run(pkt(0))
    p.close()
    assert all(n.startswith("zap-perception") for n in fake.threads)


def test_closed_perception_refuses_to_run():
    fake = Fake()
    p = make(fake)
    p.close()
    p.close()  # idempotent
    assert fake.closed == 3  # one fake stands in for all three tasks
    with pytest.raises(RuntimeError):
        p.run(pkt(0))


def test_pick_physical_skips_virtual_cameras():
    assert pick_physical([(0, ""), (1, "DemoCreator Virtual Camera"), (2, "OBS Virtual Camera")]) == 0
    assert pick_physical([(0, "OBS Virtual Camera"), (1, "Integrated Webcam")]) == 1
    assert pick_physical([(0, "obs-cam"), (1, "Some VIRTUAL cam")]) is None
    assert pick_physical([]) is None


def test_camera_candidates_prefer_dshow_then_msmf():
    listings = {
        "DSHOW": [(0, ""), (1, "DemoCreator Virtual Camera"), (2, "OBS Virtual Camera")],
        "MSMF": [(0, "USB2.0 HD UVC WebCam")],
    }
    assert camera_candidates(listings) == [("DSHOW", 0), ("MSMF", 0)]
    assert camera_candidates({"DSHOW": [(0, "OBS Virtual Camera")], "MSMF": [(3, "Cam")]}) == [("MSMF", 3)]
    assert camera_candidates({}) == []


class NoHands(Fake):
    """Like Fake, but the hand model finds hands only when `hands_visible` is set."""

    hands_visible = False

    def recognize_for_video(self, image, ts):
        result = super().recognize_for_video(image, ts)
        if not self.hands_visible:
            result.hand_landmarks, result.handedness, result.gestures = [], [], []
        return result


def test_hand_model_runs_every_other_frame_while_no_hand_is_visible():
    fake = NoHands()
    p = make(fake, hands_idle_every=2, face_every=100, mask_every=100)
    ran = []
    for i in range(6):
        sig = p.run(pkt(i))
        ran.append("hands" in p.ran)
        assert sig.hands == []
    assert ran == [True, False, True, False, True, False]
    fake.hands_visible = True
    p.run(pkt(6))  # the idle cadence still decides this frame ...
    ran = []
    for i in range(7, 11):
        p.run(pkt(i))
        ran.append("hands" in p.ran)
    assert ran == [True, True, True, True]  # ... but once a hand is seen it runs on every frame
    p.close()


def test_need_hands_forces_the_hand_model():
    fake = NoHands()
    p = make(fake, hands_idle_every=3)
    p.run(pkt(0))
    p.run(pkt(1), need_hands=True)
    assert "hands" in p.ran
    p.run(pkt(2))
    assert "hands" not in p.ran
    p.close()
