"""'We'll Be Right Back': freeze, punch-in, caption, music length, holding while you're away."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from zap import config as zap_config
from zap.app import EffectController
from zap.effects import registry
from zap.effects.brb import AWAY_S, BACK_S, DEFAULT_S, MAX_S, OUT_S, BeRightBack
from zap.effects.compositor import Compositor
from zap.effects.thanos import Presence
from zap.signals import Face, Signals

W, H = 1280, 720
DT = 1 / 30
T0 = 500.0


def frame(shade: int = 0) -> np.ndarray:
    f = np.full((H, W, 3), (70, 90, 110), np.uint8)
    cv2.ellipse(f, (640, 330), (130, 170), 0, 0, 360, (110, 140, 200), -1, cv2.LINE_AA)
    cv2.circle(f, (590, 300), 14, (30, 30, 30), -1)  # eyes, so a punch-in is visible
    cv2.circle(f, (690, 300), 14, (30, 30, 30), -1)
    if shade:
        cv2.add(f, (shade, shade, shade, 0), dst=f)
    return f


def face(fresh: bool = True) -> Face:
    lm = np.zeros((478, 3), np.float32)
    ang = np.linspace(0, 2 * np.pi, 478, endpoint=False)
    lm[:, 0], lm[:, 1] = 640 + 130 * np.cos(ang), 330 + 170 * np.sin(ang)
    lm[234, :2], lm[454, :2], lm[168, :2] = (510, 320), (770, 320), (640, 300)
    return Face(lm, {}, None, 0.0, fresh)


class Sfx:
    def __init__(self, seconds: float = 0.0) -> None:
        self.seconds = seconds
        self.played: list[str] = []

    def play(self, key: str) -> None:
        self.played.append(key)

    def stop_all(self) -> None:
        pass

    def length(self, key: str) -> float:
        return self.seconds


def ctx_at(now: float, f: np.ndarray, sfx, with_face: bool = True):
    from zap.effects.base import FxContext

    sig = Signals(now, 0, [], face() if with_face else None, width=W, height=H)
    return FxContext(f, sig, now, None, sfx, last_face=face())


def play(fx: BeRightBack, seconds: float, sfx, face_until: float = 1e9, face_from: float = 1e9,
         live_shade: int = 40) -> list[tuple[float, np.ndarray | None]]:
    """Run the effect from T0; the live frame differs from the frozen one. Returns (t, out) while active."""
    f0, live = frame(), frame(live_shade)
    fx.start(ctx_at(T0, f0, sfx))
    out = []
    t = 0.0
    while t < seconds:
        now = T0 + t
        seen = t < face_until or t >= face_from
        img = (f0 if t == 0 else live).copy()
        if not fx.active:
            break
        fx.update(ctx_at(now, f0 if t == 0 else live, sfx, seen), img)
        out.append((t, img if fx.active else None))
        t = round(t + DT, 6)
    return out


def test_registered_and_plays_its_music():
    registry.load_builtin()
    fx = registry.create("brb")
    sfx = Sfx()
    fx.start(ctx_at(T0, frame(), sfx))
    assert isinstance(fx, BeRightBack) and sfx.played == ["brb"]


def test_freezes_punches_in_and_shows_the_caption():
    sfx = Sfx(3.0)
    fx = BeRightBack()
    still = play(fx, 1.0, sfx)[-1][1]  # ~1 s in
    live = frame(40)
    assert not np.array_equal(still, live)  # frozen, not the live picture
    # punched in on the face: the face blob covers far more of the frame than before
    skin = lambda img: np.mean(np.all(np.abs(img.astype(int) - (110, 140, 200)) < 40, axis=-1))
    assert skin(still) > 2.0 * skin(frame())
    # the yellow caption in the lower part of the frame
    yellow = (still[..., 2] > 200) & (still[..., 1] > 170) & (still[..., 0] < 80)
    assert yellow[int(0.65 * H):].sum() > 2000 and yellow[: int(0.5 * H)].sum() == 0


def test_lasts_as_long_as_the_music_then_goes_back_to_live():
    for music, expect in ((3.0, 3.0), (0.0, DEFAULT_S), (60.0, MAX_S), (0.5, 2.0)):
        fx = BeRightBack()
        run = play(fx, 20.0, Sfx(music))
        end = run[-1][0]
        assert end == pytest.approx(expect + OUT_S, abs=2 * DT), music
        assert not fx.active


def test_seconds_setting_overrides_the_music():
    fx = BeRightBack(seconds=2.5)
    assert play(fx, 20.0, Sfx(9.0))[-1][0] == pytest.approx(2.5 + OUT_S, abs=2 * DT)


def test_holds_while_you_are_away_and_ends_when_you_are_back():
    fx = BeRightBack()
    run = play(fx, 60.0, Sfx(3.0), face_until=2.0, face_from=30.0)
    end = run[-1][0]
    assert fx.state != "show" and not fx.active
    assert end == pytest.approx(30.0 + BACK_S + OUT_S, abs=3 * DT)
    held = [img for t, img in run if 5.0 < t < 29.0]
    assert all(np.array_equal(held[-1], img) for img in held[-30:])  # a still card (the zoom has settled)


def test_no_hold_when_turned_off():
    fx = BeRightBack(hold_when_away=False)
    assert play(fx, 20.0, Sfx(3.0), face_until=1.0)[-1][0] == pytest.approx(3.0 + OUT_S, abs=2 * DT)


def test_short_face_dropout_is_not_leaving():
    fx = BeRightBack()
    run = play(fx, 20.0, Sfx(3.0), face_until=2.6, face_from=2.6 + AWAY_S / 2)
    assert run[-1][0] == pytest.approx(3.0 + OUT_S, abs=2 * DT)


def test_finish_goes_back_to_live_early():
    sfx = Sfx(5.0)
    fx = BeRightBack()
    f = frame()
    fx.start(ctx_at(T0, f, sfx))
    fx.update(ctx_at(T0, f, sfx), f.copy())
    fx.finish(ctx_at(T0 + 1.0, f, sfx))
    fx.update(ctx_at(T0 + 1.0 + OUT_S + DT, f, sfx), f.copy())
    assert not fx.active


def test_works_without_a_face():
    sfx = Sfx(2.0)
    fx = BeRightBack(hold_when_away=False)
    f = frame()
    from zap.effects.base import FxContext

    ctx = FxContext(f, Signals(T0, 0, [], None, width=W, height=H), T0, None, sfx)
    fx.start(ctx)
    out = f.copy()
    fx.update(ctx, out)
    assert out.shape == f.shape


def test_controller_builds_it_from_config_and_a_second_press_ends_it():
    registry.load_builtin()
    cfg = zap_config.defaults()
    cfg.brb.seconds = 3.0
    comp = Compositor(presence=Presence())
    ctl = EffectController(cfg, comp, comp.presence)
    sfx = Sfx(9.0)
    f = frame()
    ctx = ctx_at(T0, f, sfx)
    assert ctl.trigger("point_up", ctx)
    fx = comp.find("brb")
    assert fx is not None and fx.duration == 3.0
    comp.render(ctx)
    assert ctl.trigger("point_up", ctx_at(T0 + 1.0, f, sfx))  # again: back to live
    assert comp.find("brb") is fx and fx.state == "out"


def test_frame_cost():
    fx = BeRightBack()
    sfx = Sfx(4.0)
    f = frame()
    fx.start(ctx_at(T0, f, sfx))
    out = f.copy()
    times = []
    for k in range(60):
        ctx = ctx_at(T0 + k * DT, f, sfx)
        np.copyto(out, f)
        t0 = time.perf_counter()
        fx.update(ctx, out)
        times.append((time.perf_counter() - t0) * 1e3)
    assert np.percentile(times[1:], 95) < 15.0
