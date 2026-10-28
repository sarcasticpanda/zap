"""Charge (ember sparks pre-roll) on synthetic frames: no camera, mic or Zoom."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from zap.effects.base import FxContext, Layer
from zap.effects.charge import FADE_S, MAX_S, Charge
from zap.signals import Signals

W, H = 1280, 720
DT = 1 / 30


@pytest.fixture
def frame() -> np.ndarray:
    f = np.full((H, W, 3), 90, np.uint8)
    cv2.circle(f, (900, 400), 120, (60, 120, 200), -1)
    return f


def ctx_at(frame: np.ndarray, now: float) -> FxContext:
    return FxContext(frame=frame, sig=Signals(t=now, seq=0), now=now)


def render(c: Charge, frame: np.ndarray, now: float) -> np.ndarray:
    out = frame.copy()
    c.update(ctx_at(frame, now), out)
    assert out.shape == (H, W, 3) and out.dtype == np.uint8
    return out


def glow(out: np.ndarray, frame: np.ndarray) -> int:
    return int(cv2.absdiff(out, frame).astype(np.int64).sum())


def test_runs_and_auto_ends(frame: np.ndarray) -> None:
    c = Charge(seed=0)
    assert c.layer == Layer.OVERLAY and not c.active
    c.start(ctx_at(frame, 0.0), where=(900, 400))
    assert c.active
    early, full = glow(render(c, frame, 0.05), frame), glow(render(c, frame, 0.6), frame)
    assert 0 < early < full  # intensifies
    now = 0.0
    while c.active and now < 2.0:
        now += DT
        render(c, frame, now)
    assert not c.active and MAX_S <= now < MAX_S + 2 * DT


def test_additive_and_local(frame: np.ndarray) -> None:
    c = Charge(seed=1)
    c.start(ctx_at(frame, 0.0), where=(900, 400))
    out = render(c, frame, 0.5)
    assert np.all(out >= frame)  # additive glow only brightens
    changed = np.argwhere(cv2.absdiff(out, frame).max(axis=2) > 0)
    (y0, x0), (y1, x1) = changed.min(axis=0), changed.max(axis=0)
    assert x1 - x0 < 300 and y1 - y0 < 300  # small ROI around the hand
    assert x0 <= 900 <= x1 and y0 <= 400 <= y1
    assert out[..., 2].astype(int).sum() - frame[..., 2].astype(int).sum() > \
        out[..., 0].astype(int).sum() - frame[..., 0].astype(int).sum()  # orange: more red than blue


def test_discharge_fades_out(frame: np.ndarray) -> None:
    c = Charge(seed=2)
    c.start(ctx_at(frame, 0.0), where=(900, 400))
    render(c, frame, 0.3)
    c.discharge()
    mid = glow(render(c, frame, 0.3 + FADE_S / 2), frame)
    assert c.active and mid > 0
    render(c, frame, 0.3 + FADE_S + 0.01)
    assert not c.active


def test_release_discharges(frame: np.ndarray) -> None:
    c = Charge(seed=3)
    c.start(ctx_at(frame, 0.0), where=(900, 400))
    c.release(ctx_at(frame, 0.2))
    render(c, frame, 0.2 + FADE_S + 0.01)
    assert not c.active


@pytest.mark.parametrize("where", [(0, 0), (W - 1, H - 1), (-80, 900), (W + 500, -500), None])
def test_edges_and_missing_where(frame: np.ndarray, where) -> None:
    c = Charge(seed=4)
    c.start(ctx_at(frame, 0.0), where=where)
    for i in range(10):
        render(c, frame, i * DT)
    c.follow((640, 360))
    render(c, frame, 0.4)
    c.follow(None)
    assert c.active


def test_timing(frame: np.ndarray) -> None:
    c = Charge(seed=5)
    c.start(ctx_at(frame, 0.0), where=(640, 360))
    times = []
    now = 0.0
    while c.active:
        out = frame.copy()
        t = time.perf_counter()
        c.update(ctx_at(frame, now), out)
        times.append((time.perf_counter() - t) * 1e3)
        now += DT
    p95 = float(np.percentile(times, 95))
    print(f"\ncharge per-frame p95 {p95:.2f} ms")
    assert p95 < 10.0
