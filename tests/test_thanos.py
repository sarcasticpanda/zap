"""Presence (Thanos dissolve) and BackgroundPlate on synthetic frames: no camera, mic or Zoom."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from zap.effects.base import FxContext, Layer
from zap.effects.plate import BackgroundPlate
from zap.effects.thanos import REASSEMBLE_S, Presence
from zap.signals import Signals

W, H = 1280, 720
MW, MH = 640, 360
DT = 1 / 30
VANISH_S = 5.0  # longest dissolve: burn front (~2.1 s) + the longest-lived flake (2.3 s)
CENTER, AXES = (640, 420), (230, 330)  # the "person", clipped by the bottom edge like a real one


class RecordingSfx:
    def __init__(self) -> None:
        self.played: list[str] = []

    def play(self, key: str) -> None:
        self.played.append(key)

    def stop_all(self) -> None:
        pass


def background() -> np.ndarray:
    xs = np.linspace(30, 220, W, dtype=np.float32)[None, :]
    ys = np.linspace(40, 200, H, dtype=np.float32)[:, None]
    bg = np.empty((H, W, 3), np.uint8)
    bg[..., 0] = (xs + 0 * ys).astype(np.uint8)
    bg[..., 1] = (ys + 0 * xs).astype(np.uint8)
    bg[..., 2] = ((xs + ys) / 2).astype(np.uint8)
    return bg


def person_frame(bg: np.ndarray, dx: int = 0) -> np.ndarray:
    f = bg.copy()
    cv2.ellipse(f, (CENTER[0] + dx, CENTER[1]), AXES, 0, 0, 360, (40, 90, 210), -1)
    cv2.circle(f, (CENTER[0] + dx - 60, CENTER[1] - 150), 30, (230, 230, 230), -1)  # some texture
    return f


def person_region(dx: int = 0) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (CENTER[0] + dx, CENTER[1]), AXES, 0, 0, 360, 255, -1)
    return m > 0


def low_mask(dx: int = 0) -> np.ndarray:
    m = np.zeros((MH, MW), np.float32)
    cv2.ellipse(m, ((CENTER[0] + dx) // 2, CENTER[1] // 2), (AXES[0] // 2, AXES[1] // 2), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (5, 5), 0)


class Scene:
    """Drives presence + plate the way the app does: a new FxContext per frame, simulated time."""

    def __init__(self, presence: Presence, plate: BackgroundPlate, dx: int = 0) -> None:
        self.presence, self.plate = presence, plate
        self.bg = background()
        self.sfx = RecordingSfx()
        self.now = 100.0
        self.seq = 0
        self.dx = dx
        self.times: list[float] = []

    def ctx(self, with_mask: bool = True) -> FxContext:
        self.seq += 1
        frame = person_frame(self.bg, self.dx)
        mask = low_mask(self.dx) if with_mask else None
        sig = Signals(t=self.now, seq=self.seq, mask=mask, mask_fresh=with_mask, width=W, height=H)
        return FxContext(frame=frame, sig=sig, now=self.now, sfx=self.sfx, plate=self.plate)

    def step(self, with_mask: bool = True) -> tuple[FxContext, np.ndarray]:
        self.now += DT
        ctx = self.ctx(with_mask)
        if with_mask and self.plate is not None:
            self.plate.update(ctx.frame, ctx.sig.mask, ctx.now)
        out = ctx.frame.copy()
        t = time.perf_counter()
        self.presence.update(ctx, out)
        self.times.append((time.perf_counter() - t) * 1e3)
        assert out.shape == (H, W, 3) and out.dtype == np.uint8 and out.flags.c_contiguous
        return ctx, out

    def run(self, seconds: float) -> tuple[FxContext, np.ndarray]:
        last = None
        for _ in range(round(seconds / DT)):
            last = self.step()
        return last


def clean_plate() -> BackgroundPlate:
    plate = BackgroundPlate(W, H)
    plate.capture_clean(0.2, now=0.0)
    bg = background()
    for i in range(4):
        plate.update(bg, None, now=0.05 * i)
    return plate


@pytest.fixture
def scene() -> Scene:
    return Scene(Presence(seed=1), clean_plate())


def diff_in(a: np.ndarray, b: np.ndarray, region: np.ndarray) -> float:
    return float(cv2.absdiff(a, b)[region].mean())


# ---------------------------------------------------------------- presence
def test_always_active_and_mask_need(scene: Scene) -> None:
    p = scene.presence
    assert p.layer == Layer.PRESENCE and p.active and p.state == "visible"
    assert p.needs_mask is False
    ctx, out = scene.step()
    assert np.array_equal(out, ctx.frame)  # visible: untouched
    p.abort()
    assert p.active and p.state == "visible"


def test_dissolve_to_vanished_hides_person(scene: Scene) -> None:
    p = scene.presence
    ctx = scene.ctx()
    assert p.vanish(ctx, where=(900.0, 400.0))
    assert p.state == "dissolving" and p.needs_mask
    assert scene.sfx.played == ["presence"]
    n = p._dust.n
    assert 40_000 <= n <= 90_000, n
    states = []
    for _ in range(int(VANISH_S / DT)):
        _, out = scene.step()
        states.append(p.state)
    assert states[0] == "dissolving" and states[-1] == "vanished"
    region = person_region()
    assert diff_in(out, scene.plate.image(), region) < 2.0
    assert diff_in(out, scene.bg, region) < 2.0
    outside = ~cv2.dilate(region.astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool)
    assert diff_in(out, scene.bg, outside) < 0.5  # rest of the room is the live frame
    p95 = float(np.percentile(scene.times, 95))
    print(f"\npresence dissolve: {n} flakes, setup {p.setup_ms:.1f} ms, per-frame p95 {p95:.2f} ms, "
          f"max {max(scene.times):.2f} ms")
    assert p95 < 40.0


def test_vanished_follows_moving_person(scene: Scene) -> None:
    p = scene.presence
    p.vanish(scene.ctx())
    scene.run(VANISH_S)
    assert p.state == "vanished"
    scene.dx = 150  # the user moves while invisible
    _, out = scene.step()
    assert diff_in(out, scene.bg, person_region(150)) < 2.0


def test_reappear_returns_to_live(scene: Scene) -> None:
    p = scene.presence
    p.vanish(scene.ctx())
    scene.run(VANISH_S)
    assert p.toggle(scene.ctx()) and p.state == "reassembling"
    assert scene.sfx.played == ["presence", "reappear"]
    mid = None
    for i in range(int((REASSEMBLE_S + 0.3) / DT)):
        ctx, out = scene.step()
        if i == int(1.0 / DT):
            mid = (p.state, diff_in(out, ctx.frame, person_region()))
    assert mid[0] == "reassembling" and mid[1] > 5.0  # half way: visibly incomplete
    assert p.state == "visible" and not p.needs_mask
    assert np.array_equal(out, ctx.frame)


def test_reassembly_just_before_crossfade_matches_live(scene: Scene) -> None:
    p = scene.presence
    p.vanish(scene.ctx())
    scene.run(VANISH_S)
    p.reappear(scene.ctx())
    ctx, out = scene.run(REASSEMBLE_S + 0.05)  # t has reached 0, crossfade running
    assert p.state == "reassembling"
    assert diff_in(out, ctx.frame, person_region()) < 3.0


def test_reverse_mid_dissolve(scene: Scene) -> None:
    p = scene.presence
    p.vanish(scene.ctx())
    scene.run(1.0)
    assert p.state == "dissolving"
    assert p.toggle(scene.ctx()) and p.state == "reassembling"
    for _ in range(int((REASSEMBLE_S + 0.5) / DT)):
        scene.step()
        if p.state == "visible":
            break
    assert p.state == "visible"


def test_reverse_mid_reassembly(scene: Scene) -> None:
    p = scene.presence
    p.vanish(scene.ctx())
    scene.run(VANISH_S)
    p.reappear(scene.ctx())
    scene.run(1.0)
    assert p.state == "reassembling"
    assert p.toggle(scene.ctx()) and p.state == "dissolving"
    scene.run(VANISH_S)
    assert p.state == "vanished"


def test_reset_and_noops(scene: Scene) -> None:
    p = scene.presence
    assert p.reappear(scene.ctx()) is False  # already visible
    p.vanish(scene.ctx())
    assert p.vanish(scene.ctx()) is False
    scene.run(0.5)
    p.reset()
    assert p.state == "visible" and p.active
    ctx, out = scene.step()
    assert np.array_equal(out, ctx.frame)


def test_no_mask_stays_visible(scene: Scene) -> None:
    p = scene.presence
    assert p.vanish(scene.ctx(with_mask=False)) is False
    assert p.state == "visible" and scene.sfx.played == []


def test_stride_degrade() -> None:
    a = Scene(Presence(seed=2), clean_plate())
    b = Scene(Presence(seed=2), clean_plate())
    b.presence.set_stride(3)
    a.presence.vanish(a.ctx())
    b.presence.vanish(b.ctx())
    assert b.presence._dust.n < 0.6 * a.presence._dust.n
    b.run(VANISH_S)
    assert b.presence.state == "vanished"
    with pytest.raises(ValueError):
        a.presence.set_stride(4)


def test_without_plate_uses_private_plate() -> None:
    s = Scene(Presence(seed=3), clean_plate())
    s.plate = None
    s.presence.vanish(s.ctx())
    _, out = s.run(VANISH_S)
    assert s.presence.state == "vanished"
    assert out.shape == (H, W, 3)


# ---------------------------------------------------------------- plate
def test_plate_learns_background_and_fills_holes() -> None:
    bg = background()
    frame = person_frame(bg)
    plate = BackgroundPlate(W, H)
    assert plate.coverage == 0.0 and not plate.ready
    plate.update(frame, low_mask()[..., None], now=1.0)  # (h, w, 1) like MediaPipe returns
    assert plate.ready
    assert 0.5 < plate.coverage < 0.95
    img = plate.image()
    assert img.shape == (H, W, 3) and img.dtype == np.uint8
    region = person_region()
    far = ~cv2.dilate(region.astype(np.uint8), np.ones((81, 81), np.uint8)).astype(bool)
    assert diff_in(img, bg, far) < 1.0
    # the never-seen hole is inpainted from the surroundings, not black and not the person
    assert img[region].min() > 0
    assert diff_in(img, bg, region) < diff_in(frame, bg, region) / 3


def test_plate_fallback_before_anything_seen() -> None:
    plate = BackgroundPlate(W, H)
    frame = person_frame(background())
    plate.update(frame, None, now=0.0)
    img = plate.image()
    assert not plate.ready and img.shape == (H, W, 3)
    assert diff_in(img, frame, np.ones((H, W), bool)) < 20  # a heavily blurred copy of the frame


def test_plate_capture_clean_progress() -> None:
    plate = BackgroundPlate(W, H)
    bg = background()
    plate.capture_clean(1.0, now=10.0)
    assert plate.capturing and plate.progress == 0.0
    plate.update(bg, None, now=10.5)
    assert plate.capturing and 0.4 < plate.progress < 0.6
    plate.update(bg, None, now=11.1)
    assert not plate.capturing and plate.progress == 1.0
    assert plate.coverage == 1.0
    assert diff_in(plate.image(), bg, np.ones((H, W), bool)) < 1.0


def test_plate_save_load(tmp_path) -> None:
    plate = clean_plate()
    path = tmp_path / "plates" / "room.png"
    assert plate.save(path) and path.is_file()
    bg = background()
    other = BackgroundPlate(W, H)
    assert other.load(path, current_frame=bg)
    assert other.coverage == 1.0 and diff_in(other.image(), bg, np.ones((H, W), bool)) < 1.0
    dark = (bg // 2).astype(np.uint8)
    assert not BackgroundPlate(W, H).load(path, current_frame=dark)
    assert not BackgroundPlate(W, H).save(tmp_path / "empty.png")
    assert not BackgroundPlate(W, H).load(tmp_path / "missing.png", current_frame=bg)


# ---------------------------------------------------------------- snap entry
def test_start_hidden_shows_only_the_room_then_reassembles(scene: Scene) -> None:
    scene.presence.start_hidden()
    assert scene.presence.state == "vanished" and scene.presence.needs_mask and scene.presence.hidden
    _, out = scene.step()
    region = person_region()
    assert diff_in(out, scene.bg, region) < 2.0  # the meeting sees the room, not the person
    assert scene.sfx.played == []  # no dissolve, no sound
    ctx = scene.ctx()
    assert scene.presence.reappear(ctx, where=(700, 300))
    assert scene.presence.state == "reassembling" and scene.sfx.played == ["reappear"]
    _, out = scene.run(REASSEMBLE_S + 0.4)
    assert scene.presence.state == "visible"
    assert diff_in(out, person_frame(scene.bg), region) < 1.0
