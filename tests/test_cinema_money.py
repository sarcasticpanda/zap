"""Cinema and money effects run through their whole timelines on synthetic frames (no camera, mic or Zoom)."""

from __future__ import annotations

import itertools
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from zap.effects import cinema, money
from zap.effects.base import FxContext, Layer
from zap.effects.registry import EFFECTS, create
from zap.signals import FACE_LEFT_EDGE, FACE_NOSE_BRIDGE, FACE_RIGHT_EDGE, Face, Hand, Signals

W, H = 1280, 720
FPS = 30.0
DT = 1.0 / FPS
P95_BUDGET_MS = 40.0  # generous: the real targets are ~9 ms (cinema) and ~7 ms (money)


# ---------------------------------------------------------------- synthetic world
def synth_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 255, W, dtype=np.float32)
    ys = np.linspace(0, 255, H, dtype=np.float32)
    f = np.empty((H, W, 3), np.uint8)
    f[..., 0] = (xs[None, :] * 0.8 + 20).astype(np.uint8)
    f[..., 1] = (ys[:, None] * 0.6 + 40).astype(np.uint8)
    f[..., 2] = 160
    cv2.ellipse(f, (W // 2, int(0.55 * H)), (200, 300), 0, 0, 360, (90, 130, 210), -1)  # "person"
    noise = rng.integers(-12, 13, f.shape, dtype=np.int16)
    return np.clip(f.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def synth_mask(value: float | None = None) -> np.ndarray:
    """(360, 640, 1) float32 person confidence, the shape MediaPipe's segmenter returns."""
    if value is not None:
        return np.full((360, 640, 1), value, np.float32)
    m = np.zeros((360, 640), np.float32)
    cv2.ellipse(m, (320, int(0.55 * 360)), (100, 150), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (9, 9), 0)[..., None]


def synth_face(cx: float = W / 2, cy: float = 0.4 * H, fw: float = 220.0) -> Face:
    rng = np.random.default_rng(1)
    lm = np.zeros((478, 3), np.float32)
    lm[:, 0] = cx + rng.uniform(-fw / 2, fw / 2, 478)
    lm[:, 1] = cy + rng.uniform(-0.6 * fw, 0.7 * fw, 478)
    lm[FACE_NOSE_BRIDGE, :2] = (cx, cy)
    lm[FACE_LEFT_EDGE, :2] = (cx - fw / 2, cy)
    lm[FACE_RIGHT_EDGE, :2] = (cx + fw / 2, cy)
    return Face(lm=lm)


def synth_hand(wx: float, wy: float, s: float = 70.0) -> Hand:
    """An upright open hand: wrist at (wx, wy), fingers straight up."""
    lm = np.zeros((21, 3), np.float32)
    lm[0, :2] = (wx, wy)
    for f, dx in enumerate((-0.4, -0.1, 0.15, 0.4)):
        base = 5 + 4 * f
        for j in range(4):
            lm[base + j, :2] = (wx + dx * s, wy - s * (1.0 + 0.5 * j))
    for j in range(4):
        lm[1 + j, :2] = (wx - s * (0.5 + 0.2 * j), wy - s * 0.3 * j)
    return Hand(lm=lm, score=0.9, gesture="Open_Palm", gesture_score=0.9)


def synth_signals(seq: int, t: float, *, mask=True, face=True, hands=True) -> Signals:
    return Signals(
        t=t,
        seq=seq,
        hands=[synth_hand(0.25 * W, 0.5 * H), synth_hand(0.75 * W, 0.5 * H)] if hands else [],
        face=synth_face() if face else None,
        mask=(synth_mask() if mask is True else mask) if mask is not False else None,
        mask_fresh=mask is not False,
        width=W,
        height=H,
    )


def ctx_at(frame: np.ndarray, now: float, seq: int, **sig_kw) -> FxContext:
    return FxContext(frame=frame, sig=synth_signals(seq, now, **sig_kw), now=now)


def check_frame(out: np.ndarray) -> None:
    assert out.shape == (H, W, 3)
    assert out.dtype == np.uint8
    assert out.flags["C_CONTIGUOUS"]


def p95_ms(samples: list[float]) -> float:
    return float(np.percentile(np.asarray(samples) * 1000.0, 95))


def save_png(name: str, img: np.ndarray) -> Path:
    path = Path(tempfile.gettempdir()) / name
    cv2.imwrite(str(path), img)
    print(f"\n[render] {path}")
    return path


@pytest.fixture(scope="module")
def frame() -> np.ndarray:
    return synth_frame()


@pytest.fixture(scope="module", autouse=True)
def ready() -> None:
    assert cinema.wait_ready(W, H, "timesbd.ttf", timeout=30)
    assert money.wait_ready(H, "DEAL CLOSED", "", timeout=60)
    assert money.wait_ready(H, "DEAL CLOSED", "Paisa hi paisa hoga", timeout=60)


# ---------------------------------------------------------------- cinema
def run_cinema(eff, frame: np.ndarray, hold_s: float = 2.0, max_s: float = 15.0, release: bool = True, **sig_kw):
    """Start at t=0, hold for hold_s, release, run until inactive. Returns (frames, times, release_t, end_t)."""
    eff.start(ctx_at(frame, 0.0, 0, **sig_kw))
    released_at = None
    times: list[float] = []
    grays: list[bool] = []
    n = 0
    t = 0.0
    while eff.active and t < max_s:
        if release and released_at is None and t >= hold_s:
            eff.release(ctx_at(frame, t, n, **sig_kw))
            released_at = t
        ctx = ctx_at(frame, t, n, **sig_kw)
        out = frame.copy()
        t0 = time.perf_counter()
        eff.update(ctx, out)
        times.append(time.perf_counter() - t0)
        check_frame(out)
        fading = (released_at is not None and t - released_at > cinema.HOLD_S) or t > cinema.MAX_S - cinema.FADE_S
        fading = fading or t < cinema.ENTER_S  # cross-fading in from colour
        if eff.active and not fading:
            grays.append(bool(np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])))
        n += 1
        t = n * DT
    return grays, times, released_at, t


def test_cinema_registered():
    assert EFFECTS["cinema"] is cinema.Cinema
    e = create("cinema", layout="left_right", font="timesbd.ttf")
    assert e.layer == Layer.LOOK and e.needs_mask and e.needs_face and e.sfx_key == "cinema"


def test_cinema_full_timeline_left_right(frame):
    eff = cinema.Cinema(layout="left_right")
    grays, times, released_at, end_t = run_cinema(eff, frame)
    assert not eff.active
    assert all(grays) and len(grays) > 60, "the held look must be pure grayscale"
    # release -> 0.5 s hold -> 0.3 s fade, then off (allow a frame of slack)
    assert released_at == pytest.approx(2.0, abs=DT)
    assert end_t - released_at == pytest.approx(cinema.HOLD_S + cinema.FADE_S, abs=2 * DT)
    p95 = p95_ms(times[3:])
    print(f"\n[cinema left_right] frames={len(times)} p50={np.median(times) * 1e3:.2f} ms p95={p95:.2f} ms")
    assert p95 < P95_BUDGET_MS


def test_cinema_top_bottom_and_no_perception(frame):
    eff = cinema.Cinema(layout="top_bottom")
    grays, times, _, _ = run_cinema(eff, frame, hold_s=1.0, mask=False, face=False, hands=False)
    assert all(grays) and not eff.active
    print(f"\n[cinema top_bottom, no mask/face/hands] p95={p95_ms(times[3:]):.2f} ms")


def test_cinema_unknown_layout_falls_back():
    assert cinema.Cinema(layout="diagonal").layout == "left_right"


def test_cinema_auto_ends_after_max(frame):
    eff = cinema.Cinema()
    _, _, released_at, end_t = run_cinema(eff, frame, release=False, max_s=20.0)
    assert released_at is None and not eff.active
    assert end_t == pytest.approx(cinema.MAX_S, abs=2 * DT)


def test_cinema_rehold_during_hold_keeps_going(frame):
    eff = cinema.Cinema()
    eff.start(ctx_at(frame, 0.0, 0))
    eff.update(ctx_at(frame, 0.5, 1), frame.copy())
    eff.release(ctx_at(frame, 1.0, 2))
    eff.update(ctx_at(frame, 1.2, 3), frame.copy())
    eff.start(ctx_at(frame, 1.3, 4))  # hands back up within the hold
    assert eff.active and not eff.released and eff.t0 == 0.0
    eff.update(ctx_at(frame, 3.0, 5), frame.copy())
    assert eff.active


def test_cinema_title_and_darkened_background(frame):
    eff = cinema.Cinema(layout="left_right")
    eff.start(ctx_at(frame, 0.0, 0))
    out = None
    for n in range(int(1.0 * FPS)):
        out = frame.copy()
        eff.update(ctx_at(frame, n * DT, n), out)
    g = out[..., 0]
    level = int(0.5 * H)  # wrists sit at 0.5 H, inside the 35-60% clamp
    band = slice(level - 40, level + 40)
    assert g[band, int(0.2 * W) - 120 : int(0.2 * W) + 120].max() >= 235, "ABSOLUTE should be drawn"
    assert g[band, int(0.8 * W) - 120 : int(0.8 * W) + 120].max() >= 235, "CINEMA should be drawn"
    person = g[int(0.45 * H) : int(0.55 * H), W // 2 - 60 : W // 2 + 60].mean()
    background = g[int(0.85 * H) :, 20:200].mean()
    assert person > background + 30, (person, background)
    save_png("zap_cinema.png", out)


def test_cinema_camera_scale_curve():
    assert cinema.camera_scale(0.0) == pytest.approx(1.0)
    ts = np.linspace(0, cinema.PUNCH_S, 50)
    scales = [cinema.camera_scale(t) for t in ts]
    assert all(b >= a - 1e-9 for a, b in itertools.pairwise(scales))  # smooth: no overshoot, never backs up
    assert cinema.camera_scale(cinema.PUNCH_S) == pytest.approx(cinema.PUNCH_SCALE)
    assert cinema.camera_scale(100.0) == pytest.approx(cinema.PUSH_SCALE)


# ---------------------------------------------------------------- money
def run_money(eff, frame: np.ndarray, max_s: float = 15.0, **sig_kw):
    eff.start(ctx_at(frame, 0.0, 0, **sig_kw))
    times: list[float] = []
    counts: list[int] = []
    changed = 0
    n = 0
    t = 0.0
    snapshot = None
    while eff.active and t < max_s:
        ctx = ctx_at(frame, t, n, **sig_kw)
        out = frame.copy()
        t0 = time.perf_counter()
        eff.update(ctx, out)
        times.append(time.perf_counter() - t0)
        check_frame(out)
        counts.append(eff.alive_count)
        changed += not np.array_equal(out, frame)
        if snapshot is None and t >= 1.2:
            snapshot = out.copy()
        n += 1
        t = n * DT
    return times, counts, changed, t, snapshot


def test_money_registered():
    assert EFFECTS["money"] is money.MoneyRain
    e = create("money", caption="Paisa hi paisa hoga", stamp_text="DEAL CLOSED")
    assert e.layer == Layer.OVERLAY and e.needs_mask and e.sfx_key == "money"


def test_money_full_timeline_finishes_by_itself(frame):
    eff = money.MoneyRain(caption="Paisa hi paisa hoga", seed=3)
    times, counts, changed, end_t, snapshot = run_money(eff, frame)
    assert not eff.active
    assert end_t < money.MAX_S, "must end because every bill left, not on the hard stop"
    assert 5.0 <= end_t <= 7.0, end_t
    assert max(counts) >= 40 and counts[-1] == 0
    assert changed > 0.8 * len(times)
    busy = [tm for tm, c in zip(times, counts) if c >= 60]
    p95_all, p95_busy = p95_ms(times[1:]), p95_ms(busy)
    print(
        f"\n[money] frames={len(times)} end={end_t:.2f}s peak_bills={max(counts)} "
        f"p50={np.median(times) * 1e3:.2f} ms p95={p95_all:.2f} ms p95(>=60 bills)={p95_busy:.2f} ms"
    )
    assert p95_all < P95_BUDGET_MS and p95_busy < P95_BUDGET_MS
    save_png("zap_money.png", snapshot)


def test_money_behind_bills_hidden_by_person(frame):
    # every bill behind, the whole frame is "person", no stamp: the picture must come out untouched
    eff = money.MoneyRain(stamp_text="", caption="", behind_ratio=1.0, seed=5)
    _times, counts, changed, _end_t, _ = run_money(eff, frame, mask=synth_mask(1.0))
    assert changed == 0 and max(counts) > 0 and not eff.active


def test_money_behind_bills_visible_without_person(frame):
    eff = money.MoneyRain(stamp_text="", caption="", behind_ratio=1.0, seed=5)
    _, _, changed, _, _ = run_money(eff, frame, mask=synth_mask(0.0))
    assert changed > 0


def test_money_without_mask(frame):
    eff = money.MoneyRain(seed=7)
    _times, _, _, end_t, _ = run_money(eff, frame, mask=False)
    assert not eff.active and end_t < money.MAX_S


def test_money_cold_start_never_stalls():
    # a size nobody prewarmed: the first frames must not wait for the atlas build
    w, h = 854, 480
    f = cv2.resize(synth_frame(2), (w, h))
    eff = money.MoneyRain(seed=9)
    sig = Signals(t=0.0, seq=0, mask=synth_mask(), width=w, height=h)
    eff.start(FxContext(frame=f, sig=sig, now=0.0))
    worst = 0.0
    for n in range(10):
        out = f.copy()
        t0 = time.perf_counter()
        eff.update(FxContext(frame=f, sig=sig, now=n * DT), out)
        worst = max(worst, time.perf_counter() - t0)
        assert out.shape == (h, w, 3) and out.dtype == np.uint8
    print(f"\n[money cold start 854x480] worst first-10-frame update={worst * 1e3:.2f} ms")
    assert worst < 0.25
    assert money.wait_ready(h, "DEAL CLOSED", "", timeout=60)


def test_money_atlas_is_complete():
    atlas = money._atlas(H)
    assert atlas is not None and atlas.complete
    assert atlas.lut.shape == (len(money.VARIANTS), 2, len(money.TILTS), money.N_FLIP)
    assert atlas.lut.max() < len(atlas.pm) == len(atlas.inv) == len(atlas.half)
    bill_w = [atlas.pm[atlas.lut[0, d, len(money.TILTS) // 2, 0]].shape[1] for d in (0, 1)]
    assert 100 <= bill_w[0] < bill_w[1] <= 160, bill_w  # untilted, face-on bills at 720p


def test_money_mask_roi_matches_full_upsample():
    from zap.render.blend import upsample_mask

    low = synth_mask()[..., 0]
    full = upsample_mask(low, W, H)
    for x0, y0, w, h in ((0, 0, 150, 70), (500, 300, 140, 160), (1180, 650, 100, 70), (613, 41, 37, 23)):
        roi = money.mask_roi(low, x0, y0, w, h, W, H)
        assert roi.shape == (h, w) and roi.dtype == np.float32
        assert np.abs(roi - full[y0 : y0 + h, x0 : x0 + w]).max() < 2e-3
