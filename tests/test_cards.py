"""Freeze cards (We'll Be Right Back, To Be Continued) and the snap-twice leave, through the EffectController."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from zap import config as zap_config
from zap.app import EffectController
from zap.effects import registry
from zap.effects.base import FxContext
from zap.effects.cards import AWAY_S, BACK_S, MAX_S, OUT_S, BeRightBack, ToBeContinued
from zap.effects.compositor import Compositor
from zap.gestures.types import CANCEL, SNAP1, SNAP2, Intent
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


def ctx_at(now: float, f: np.ndarray, sfx, with_face: bool = True) -> FxContext:
    sig = Signals(now, 0, [], face() if with_face else None, width=W, height=H)
    return FxContext(f, sig, now, None, sfx, last_face=face())


def play(fx, seconds: float, sfx, face_until: float = 1e9, face_from: float = 1e9, live_shade: int = 40):
    """Run the card from T0; the live frame differs from the frozen one. Returns (t, out) per frame."""
    f0, live = frame(), frame(live_shade)
    fx.start(ctx_at(T0, f0, sfx))
    out = []
    t = 0.0
    while t < seconds and fx.active:
        seen = t < face_until or t >= face_from
        img = (f0 if t == 0 else live).copy()
        fx.update(ctx_at(T0 + t, f0 if t == 0 else live, sfx, seen), img)
        out.append((t, img if fx.active else None))
        t = round(t + DT, 6)
    return out


# ---------------------------------------------------------------- We'll Be Right Back
def test_registered_and_play_their_music():
    registry.load_builtin()
    for name, cls in (("brb", BeRightBack), ("tbc", ToBeContinued)):
        fx, sfx = registry.create(name), Sfx()
        fx.start(ctx_at(T0, frame(), sfx))
        assert isinstance(fx, cls) and sfx.played == [name]


def test_brb_freezes_punches_in_and_shows_the_caption():
    still = play(BeRightBack(), 1.0, Sfx(3.0))[-1][1]  # ~1 s in
    assert not np.array_equal(still, frame(40))  # frozen, not the live picture
    skin = lambda img: np.mean(np.all(np.abs(img.astype(int) - (110, 140, 200)) < 40, axis=-1))
    assert skin(still) > 2.0 * skin(frame())  # punched in on the face
    yellow = (still[..., 2] > 200) & (still[..., 1] > 170) & (still[..., 0] < 80)
    assert yellow[int(0.65 * H):].sum() > 2000 and yellow[: int(0.5 * H)].sum() == 0


def test_cards_last_as_long_as_the_music_then_go_back_to_live():
    for cls, music, expect in ((BeRightBack, 3.0, 3.0), (BeRightBack, 0.0, 4.5), (BeRightBack, 60.0, MAX_S),
                               (BeRightBack, 0.5, 2.0), (ToBeContinued, 1.5, 3.0), (ToBeContinued, 6.0, 6.0)):
        fx = cls()
        run = play(fx, 20.0, Sfx(music))
        assert run[-1][0] == pytest.approx(expect + OUT_S, abs=2 * DT), (cls.__name__, music)
        assert not fx.active


def test_seconds_setting_overrides_the_music():
    assert play(BeRightBack(seconds=2.5), 20.0, Sfx(9.0))[-1][0] == pytest.approx(2.5 + OUT_S, abs=2 * DT)


def test_brb_holds_while_you_are_away_and_ends_when_you_are_back():
    fx = BeRightBack()
    run = play(fx, 60.0, Sfx(3.0), face_until=2.0, face_from=30.0)
    assert not fx.active and run[-1][0] == pytest.approx(30.0 + BACK_S + OUT_S, abs=3 * DT)
    held = [img for t, img in run if 5.0 < t < 29.0]
    assert all(np.array_equal(held[-1], img) for img in held[-30:])  # a still card (the zoom has settled)


def test_tbc_does_not_wait_for_you():
    assert play(ToBeContinued(), 20.0, Sfx(3.0), face_until=1.0)[-1][0] == pytest.approx(3.0 + OUT_S, abs=2 * DT)


def test_short_face_dropout_is_not_leaving():
    run = play(BeRightBack(), 20.0, Sfx(3.0), face_until=2.6, face_from=2.6 + AWAY_S / 2)
    assert run[-1][0] == pytest.approx(3.0 + OUT_S, abs=2 * DT)


def test_works_without_a_face():
    for cls in (BeRightBack, ToBeContinued):
        f = frame()
        ctx = FxContext(f, Signals(T0, 0, [], None, width=W, height=H), T0, None, Sfx(2.0))
        fx = cls()
        fx.start(ctx)
        out = f.copy()
        fx.update(ctx, out)
        assert out.shape == f.shape


def test_frame_cost():
    for cls in (BeRightBack, ToBeContinued):
        fx, sfx, f = cls(), Sfx(4.0), frame()
        fx.start(ctx_at(T0, f, sfx))
        out, times = f.copy(), []
        for k in range(60):
            np.copyto(out, f)
            t0 = time.perf_counter()
            fx.update(ctx_at(T0 + k * DT, f, sfx), out)
            times.append((time.perf_counter() - t0) * 1e3)
        assert np.percentile(times[1:], 95) < 15.0, cls.__name__


# ---------------------------------------------------------------- the controller
class FakePresence:
    """Records what the controller asks of the Thanos effect (the real one needs a person mask)."""

    def __init__(self, state: str = "visible") -> None:
        self.state = state
        self.calls: list[str] = []

    @property
    def hidden(self) -> bool:
        return self.state != "visible"

    def vanish(self, ctx, where=None) -> bool:
        self.calls.append("vanish")
        if self.state in ("dissolving", "vanished"):
            return False
        self.state = "dissolving"
        return True

    def reappear(self, ctx, where=None) -> bool:
        self.calls.append("reappear")
        if self.state in ("visible", "reassembling"):
            return False
        self.state = "reassembling"
        return True

    def toggle(self, ctx, where=None) -> bool:
        return self.vanish(ctx, where) if self.state in ("visible", "reassembling") else self.reappear(ctx, where)

    def start_hidden(self) -> None:
        self.state = "vanished"

    def reset(self) -> None:
        self.state = "visible"


def controller(presence=None, **cfg_changes):
    registry.load_builtin()
    cfg = zap_config.defaults()
    for k, v in cfg_changes.items():
        setattr(cfg.zoom, k, v)
    left: list[bool] = []
    ctl = EffectController(cfg, Compositor(), presence or FakePresence(),
                           leave=lambda end_for_all, dry_run: left.append(dry_run) or _Res(dry_run))
    return ctl, left


class _Res:
    def __init__(self, dry: bool) -> None:
        self.ok, self.message, self.dry_run = True, "left", dry


def test_open_palm_and_a_second_press_end_the_cards():
    ctl, _ = controller()
    sfx, f = Sfx(9.0), frame()
    for sign, card in (("peace", "brb"), ("l_shape", "tbc")):
        assert ctl.trigger(sign, ctx_at(T0, f, sfx))
        fx = ctl.compositor.find(card)
        assert fx is not None and ctl.card is fx and ctl.card_active
        ctl.compositor.render(ctx_at(T0, f, sfx))
        ctl.handle(Intent("cancel", CANCEL, T0 + 1.5), ctx_at(T0 + 1.5, f, sfx))  # open palm
        assert fx.state == "out" and not ctl.card_active
    ctl.trigger("peace", ctx_at(T0 + 5, f, sfx))
    ctl.trigger("peace", ctx_at(T0 + 6, f, sfx))  # the hotkey again
    assert ctl.compositor.find("brb").state == "out"


def test_snap_twice_vanishes_then_leaves():
    ctl, left = controller(countdown_s=3.0, dry_run=True)
    pres, f, sfx = ctl.presence, frame(), Sfx()
    ctl.handle(Intent("charge", "", T0), ctx_at(T0, f, sfx))  # snap 1: you vanish at once
    assert pres.state == "dissolving"
    ctl.handle(Intent("trigger", SNAP2, T0 + 0.8), ctx_at(T0 + 0.8, f, sfx))  # snap 2
    assert pres.state == "dissolving" and pres.calls == ["vanish"]  # stays gone, no flicker back
    assert ctl.leaving is not None and ctl.modal_active
    for k in range(int(3.5 / DT)):
        ctl.compositor.render(ctx_at(T0 + 0.8 + k * DT, f, sfx))
    for th in ctl.leave_threads:
        th.join(2.0)
    assert left == [True]  # left the meeting (dry run)


def test_snap_twice_while_vanished_goes_and_stays_gone():
    ctl, _ = controller()
    ctl.presence.state = "vanished"
    f, sfx = frame(), Sfx()
    ctl.handle(Intent("charge", "", T0), ctx_at(T0, f, sfx))  # snap 1 starts bringing you back ...
    assert ctl.presence.state == "reassembling"
    ctl.handle(Intent("trigger", SNAP2, T0 + 0.8), ctx_at(T0 + 0.8, f, sfx))  # ... snap 2 sends you away again
    assert ctl.presence.state == "dissolving" and ctl.leaving is not None


def test_cancelling_the_leave_brings_you_back():
    ctl, left = controller(countdown_s=3.0)
    f, sfx = frame(), Sfx()
    ctl.trigger(SNAP2, ctx_at(T0, f, sfx))  # Ctrl+Alt+End: no first snap
    assert ctl.presence.state == "dissolving" and ctl.modal_active
    ctl.handle(Intent("cancel", SNAP1, T0 + 2.0), ctx_at(T0 + 2.0, f, sfx))
    assert not ctl.modal_active and ctl.presence.state == "reassembling"
    for k in range(int(3.0 / DT)):
        ctl.compositor.render(ctx_at(T0 + 2.0 + k * DT, f, sfx))
    assert left == []


def test_one_snap_still_just_vanishes():
    ctl, _ = controller()
    f, sfx = frame(), Sfx()
    ctl.handle(Intent("charge", "", T0), ctx_at(T0, f, sfx))
    ctl.handle(Intent("trigger", SNAP1, T0 + 1.0), ctx_at(T0 + 1.0, f, sfx))
    assert ctl.presence.state == "dissolving" and ctl.leaving is None
