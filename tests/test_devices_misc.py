"""Hotkeys, VirtualCam and doctor helpers with fakes: no keyboard hook, no OBS."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from zap import doctor
from zap.devices import hotkeys, vcam
from zap.devices.hotkeys import BINDINGS, HOTKEY_HELP, Hotkeys

# ---------------------------------------------------------------- hotkeys


def test_bindings_are_ctrl_alt_only_and_documented():
    assert all(combo.startswith("<ctrl>+<alt>+") for combo in BINDINGS)
    triggers = {c.split(":", 1)[1] for c in BINDINGS.values() if c.startswith("trigger:")}
    assert triggers == {"snap1", "snap2", "two_thumbs_up", "hands_up", "cover_face", "forehead", "eyebrows_up",
                        "collar", "peace", "l_shape"}
    assert len(HOTKEY_HELP) == len(BINDINGS) - 1  # Up and Down share a help line
    assert all(k.startswith("Ctrl+Alt+") for k, _ in HOTKEY_HELP)


def test_bindings_parse_with_pynput():
    from pynput.keyboard import HotKey

    for combo in BINDINGS:
        assert len(HotKey.parse(combo)) == 3


def test_hotkeys_queue_commands(monkeypatch):
    made = {}

    class FakeListener:
        daemon = False

        def __init__(self, callbacks):
            made.update(callbacks)
            self.alive = False

        def start(self):
            self.alive = True

        def wait(self):
            pass

        def is_alive(self):
            return self.alive

        def stop(self):
            self.alive = False

    monkeypatch.setattr(hotkeys, "_make_listener", FakeListener)
    hk = Hotkeys()
    assert hk.start() and hk.running
    made["<ctrl>+<alt>+p"]()  # pynput's hook thread would call these
    made["<ctrl>+<alt>+1"]()
    hk.push("reload")
    assert hk.drain() == ["panic", "trigger:snap1", "reload"]
    assert hk.drain() == []
    hk.stop()
    assert not hk.running


def test_hotkeys_tolerate_listener_failure(monkeypatch):
    def boom(callbacks):
        raise OSError("no keyboard hook")

    monkeypatch.setattr(hotkeys, "_make_listener", boom)
    hk = Hotkeys()
    assert hk.start() is False
    assert "no keyboard hook" in hk.error and not hk.running
    hk.push("panic")
    assert hk.drain() == ["panic"]
    hk.stop()


# ---------------------------------------------------------------- vcam


def fake_pyvirtualcam(monkeypatch, error: str | None = None):
    sent = []

    class Camera:
        def __init__(self, w, h, fps, fmt=None, backend=None):
            if error:
                raise RuntimeError(error)
            assert backend == "obs" and fmt == "BGR"
            self.device = "OBS Virtual Camera"
            self.frames_sent = 0

        def send(self, frame):
            assert frame.flags.c_contiguous and frame.dtype == np.uint8
            sent.append(frame.shape)
            self.frames_sent += 1

        def close(self):
            sent.append("closed")

    mod = types.ModuleType("pyvirtualcam")
    mod.Camera = Camera
    mod.PixelFormat = types.SimpleNamespace(BGR="BGR")
    monkeypatch.setitem(sys.modules, "pyvirtualcam", mod)
    return sent


@pytest.mark.parametrize("raw, friendly", [
    ("OBS Virtual Camera device not found!", vcam.NOT_INSTALLED),
    ("virtual camera output could not be started", vcam.BUSY),
])
def test_vcam_friendly_errors(monkeypatch, raw, friendly):
    fake_pyvirtualcam(monkeypatch, raw)
    with pytest.raises(vcam.VirtualCamError) as e:
        vcam.VirtualCam(1280, 720, 30)
    assert str(e.value) == friendly


def test_vcam_send_makes_frames_contiguous(monkeypatch):
    sent = fake_pyvirtualcam(monkeypatch)
    with vcam.VirtualCam(64, 36, 30) as cam:
        assert cam.device == "OBS Virtual Camera"
        big = np.zeros((36, 128, 3), np.uint8)
        cam.send(big[:, ::2])  # a strided view
        with pytest.raises(ValueError):
            cam.send(np.zeros((720, 1280, 3), np.uint8))
        assert cam.frames_sent == 1
    assert sent == [(36, 64, 3), "closed"]
    with pytest.raises(vcam.VirtualCamError):
        cam.send(np.zeros((36, 64, 3), np.uint8))


# ---------------------------------------------------------------- doctor


def test_cv2_provider_check():
    assert doctor.cv2_provider_check(["opencv-contrib-python"]).status == "OK"
    bad = doctor.cv2_provider_check(["opencv-contrib-python", "opencv-python"])
    assert bad.status == "FAIL" and "opencv-python" in bad.hint
    assert doctor.cv2_provider_check([]).status == "FAIL"
    assert doctor.cv2_provider_check(["opencv-python-headless"]).status == "FAIL"


def test_format_check_shows_fix_only_when_needed():
    assert "fix" not in doctor.format_check(doctor.Check("OK", "fine", "hint"))
    assert "fix: do this" in doctor.format_check(doctor.Check("FAIL", "broken", "do this"))


def test_run_exit_code(monkeypatch):
    lines = []
    monkeypatch.setattr(doctor, "collect", lambda: [doctor.Check("OK", "a"), doctor.Check("WARN", "b", "c")])
    assert doctor.run(out=lines.append) == 0
    assert any("Mirror my video" in s for s in lines) and any("hand gestures" in s for s in lines)
    monkeypatch.setattr(doctor, "collect", lambda: [doctor.Check("FAIL", "x", "y")])
    assert doctor.run(out=lines.append) == 1


def test_process_names_sees_python():
    names = doctor._process_names()
    assert any(n.startswith("python") for n in names)
