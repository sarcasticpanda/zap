"""ZAP on macOS and Linux: camera listing, the virtual camera, fonts, the data folder, the instance lock."""

from __future__ import annotations

import sys
import types

import cv2
import pytest

from zap import doctor, paths
from zap.devices import camera, vcam


def fake_enumerate(monkeypatch, cams, seen):
    mod = types.ModuleType("cv2_enumerate_cameras")
    mod.enumerate_cameras = lambda backend: seen.append(backend) or cams
    monkeypatch.setitem(sys.modules, "cv2_enumerate_cameras", mod)


# ---------------------------------------------------------------- the webcam
def test_linux_lists_v4l2_cameras_and_never_its_own_output(monkeypatch):
    monkeypatch.setattr(camera, "PLATFORM", "linux")
    seen = []
    fake_enumerate(monkeypatch, [types.SimpleNamespace(index=2, name="ZAP Camera"),
                                 types.SimpleNamespace(index=0, name="Integrated Webcam"),
                                 types.SimpleNamespace(index=4, name="")], seen)
    cams = camera.list_cameras()
    assert seen == [cv2.CAP_V4L2]
    assert [c.index for c in cams] == [0, 2, 4] and cams[2].name == "Camera 4"
    assert {c.backend_name for c in cams} == {"V4L2"}
    assert camera.pick_camera("", cams).name == "Integrated Webcam"
    with pytest.raises(camera.CameraError):
        camera.pick_camera("ZAP Camera", cams)  # that is ZAP's own output
    assert camera.is_self_capture("Dummy video device (0x0000)")  # v4l2loopback's default name


def test_macos_uses_avfoundation_without_the_windows_fallback(monkeypatch):
    monkeypatch.setattr(camera, "PLATFORM", "darwin")
    tried = []
    monkeypatch.setattr(camera, "pick_msmf_camera", lambda *a: tried.append("msmf"))
    monkeypatch.setattr(camera, "pick_camera", lambda pref: camera.CameraInfo(0, "FaceTime HD Camera",
                                                                            cv2.CAP_AVFOUNDATION))
    out = camera.Camera()._candidates()
    assert [c.backend_name for c in out] == ["AVFoundation"] and tried == []


# ---------------------------------------------------------------- the virtual camera
def test_linux_finds_the_device_labelled_zap_camera(tmp_path):
    for node, label in (("video0", "Integrated Webcam"), ("video4", "OBS Virtual Camera"), ("video7", "ZAP Camera")):
        (tmp_path / node).mkdir()
        (tmp_path / node / "name").write_text(label + "\n", encoding="utf-8")
    assert vcam.linux_device(sysfs=tmp_path) == "/dev/video7"
    assert vcam.linux_device("Nope", sysfs=tmp_path) is None


def test_virtual_camera_backend_per_os(monkeypatch):
    made = []

    class FakeCam:
        device, frames_sent = "dev", 0

        def __init__(self, *a, **kw):
            if kw.get("backend") == "broken":
                raise KeyError("broken")
            made.append(kw)

        def close(self):
            pass

    mod = types.ModuleType("pyvirtualcam")
    mod.Camera, mod.PixelFormat = FakeCam, types.SimpleNamespace(BGR="bgr")
    monkeypatch.setitem(sys.modules, "pyvirtualcam", mod)
    for plat in ("win32", "darwin"):
        monkeypatch.setattr(vcam, "PLATFORM", plat)
        vcam.VirtualCam().close()
    monkeypatch.setattr(vcam, "PLATFORM", "linux")
    monkeypatch.setattr(vcam, "linux_device", lambda: "/dev/video7")
    vcam.VirtualCam().close()
    assert [m["backend"] for m in made] == ["obs", "obs", "v4l2loopback"] and made[2]["device"] == "/dev/video7"


def test_virtual_camera_errors_say_what_to_do(monkeypatch):
    monkeypatch.setattr(vcam, "PLATFORM", "linux")
    assert vcam.MODPROBE in vcam.friendly_error(RuntimeError("No v4l2 loopback device found at /dev/video[0-99]"))
    assert "usermod" in vcam.friendly_error(PermissionError("Permission denied: '/dev/video7'"))
    monkeypatch.setattr(vcam, "PLATFORM", "darwin")
    assert vcam.friendly_error(RuntimeError("... not installed [Device not found]")) == vcam.NOT_INSTALLED_MAC
    monkeypatch.setattr(vcam, "PLATFORM", "win32")
    assert vcam.friendly_error(RuntimeError("device not found")) == vcam.NOT_INSTALLED


# ---------------------------------------------------------------- files and fonts
def test_windows_font_names_find_their_look_alikes(monkeypatch, tmp_path):
    serif = tmp_path / "LiberationSerif-Bold.ttf"
    serif.write_bytes(b"font")
    monkeypatch.setattr(paths, "_system_fonts", lambda: {"liberationserif-bold.ttf": serif})
    assert paths.font_path("timesbd.ttf") == serif
    assert paths.font_path("impact.ttf") is None  # nothing like it here: the caller falls back to Pillow's font


def test_data_folder_per_os(monkeypatch, tmp_path):
    monkeypatch.delenv("ZAP_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert paths.home().parts[-3:] == ("Library", "Application Support", "ZAP")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert paths.home() == tmp_path / "ZAP"
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert paths.home() == tmp_path / "Roaming" / "ZAP"


@pytest.mark.skipif(sys.platform == "win32", reason="the lock file is the macOS/Linux way (Windows: a mutex)")
def test_a_second_zap_is_refused(monkeypatch, tmp_path):
    from zap.app import AlreadyRunning, acquire_single_instance, release_single_instance

    monkeypatch.setenv("ZAP_HOME", str(tmp_path))
    first = acquire_single_instance()
    with pytest.raises(AlreadyRunning):
        acquire_single_instance()
    release_single_instance(first)
    release_single_instance(acquire_single_instance())  # free again


# ---------------------------------------------------------------- zap doctor on Linux
def test_doctor_checks_the_linux_virtual_camera(monkeypatch):
    monkeypatch.setattr(doctor.os.path, "isdir", lambda p: False)
    out = doctor.check_loopback()
    assert out[0].status == "FAIL" and vcam.MODPROBE in out[0].hint

    monkeypatch.setattr(doctor.os.path, "isdir", lambda p: True)
    monkeypatch.setattr(vcam, "linux_device", lambda: "/dev/video7")
    monkeypatch.setattr(doctor.os, "access", lambda p, mode: False)
    out = doctor.check_loopback()
    assert [c.status for c in out][:3] == ["OK", "WARN", "FAIL"]  # found; no exclusive_caps; no permission
    assert "usermod" in out[2].hint


# ---------------------------------------------------------------- setting up the call apps, per OS
def test_call_app_steps_fit_each_os():
    from zap.calls import guide

    every = [*guide.EVERY_APP, *(s for steps in guide.SETUP.values() for s in steps), *guide.NOT_SUPPORTED]
    win, mac, linux = (guide.steps(every, p) for p in ("win32", "darwin", "linux"))
    assert any("Alt+Q" in s for s in win) and any("VB-CABLE" in s for s in win)
    for elsewhere in (mac, linux):  # leaving by gesture and sound into the call are Windows-only for now
        assert not any("Snap twice leaves" in s or "VB-CABLE" in s or "DirectShow" in s for s in elsewhere)
        assert any("hang up yourself" in s for s in elsewhere)
    assert not any("hang up yourself" in s for s in win)
    assert any("ZAP Camera" in s for s in linux) and not any("OBS Virtual Camera" in s for s in linux)
    assert any("OBS Virtual Camera" in s for s in mac)
    assert not any(s.startswith("[") for s in win + mac + linux)  # the markers never show


def test_call_apps_found_on_macos_and_linux(monkeypatch, tmp_path):
    from zap.calls import guide

    (tmp_path / "zoom.us.app").mkdir()
    (tmp_path / "Discord.app").mkdir()
    monkeypatch.setattr(guide, "MAC_APPS", tmp_path)
    assert guide.installed("darwin") == ["zoom", "meet", "discord", "browser"]
    monkeypatch.setattr(guide.shutil, "which", lambda c: f"/usr/bin/{c}" if c in ("firefox", "signal-desktop") else None)
    assert guide.installed("linux") == ["meet", "signal", "browser"]
