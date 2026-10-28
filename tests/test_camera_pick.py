"""Camera selection and the capture thread, with fake devices (no webcam opened)."""

from __future__ import annotations

import sys
import threading
import time
import types
from typing import ClassVar

import cv2
import numpy as np
import pytest

from zap.devices import camera
from zap.devices.camera import Camera, CameraError, CameraInfo, is_virtual, list_cameras, pick_camera

# The measured DirectShow list on the dev machine (the real webcam has an empty name).
THIS_PC = [CameraInfo(0, "Camera 0"), CameraInfo(1, "DemoCreator Virtual Camera"), CameraInfo(2, "OBS Virtual Camera")]


@pytest.mark.parametrize("name", [
    "OBS Virtual Camera", "obs-camera", "DemoCreator Virtual Camera", "ZAP Camera", "Unity Video Capture",
    "XSplit VCam", "ManyCam Virtual Webcam", "Snap Camera", "Camo", "SoftCam",
])
def test_virtual_names(name):
    assert is_virtual(name)


@pytest.mark.parametrize("name", ["Camera 0", "USB2.0 HD UVC WebCam", "Integrated Camera", "Logitech BRIO"])
def test_physical_names(name):
    assert not is_virtual(name)


def test_auto_pick_skips_virtual_cameras():
    assert pick_camera("", THIS_PC).index == 0
    reordered = [THIS_PC[2], THIS_PC[1], CameraInfo(5, "Logitech C920")]
    assert pick_camera("", reordered).index == 5


def test_preferred_by_name_and_index():
    assert pick_camera("democreator", THIS_PC).index == 1  # virtual, but not our own output
    assert pick_camera("1", THIS_PC).index == 1
    assert pick_camera("Camera 0", THIS_PC).index == 0
    assert pick_camera("#0", THIS_PC).index == 0


@pytest.mark.parametrize("pref", ["OBS", "obs virtual camera", "2", "Camera 2"])
def test_preferred_obs_is_refused(pref):
    with pytest.raises(CameraError, match="own output"):
        pick_camera(pref, THIS_PC)


def test_preferred_zap_is_refused():
    with pytest.raises(CameraError):
        pick_camera("zap", [CameraInfo(0, "ZAP Camera"), CameraInfo(1, "Webcam")])


def test_missing_preferred_falls_back(caplog):
    assert pick_camera("Logitech", THIS_PC).index == 0
    assert "not found" in caplog.text


def test_no_physical_camera_error_lists_cameras():
    with pytest.raises(CameraError) as e:
        pick_camera("", THIS_PC[1:])
    assert "OBS Virtual Camera" in str(e.value) and "DemoCreator" in str(e.value)
    with pytest.raises(CameraError, match="no cameras found"):
        pick_camera("", [])


# ---------------------------------------------------------------- list_cameras


def fake_enum(monkeypatch, dshow, msmf):
    def enumerate_cameras(backend):
        rows = dshow if backend == cv2.CAP_DSHOW else msmf
        return [types.SimpleNamespace(index=i, name=n) for i, n in rows]

    mod = types.ModuleType("cv2_enumerate_cameras")
    mod.enumerate_cameras = enumerate_cameras
    monkeypatch.setitem(sys.modules, "cv2_enumerate_cameras", mod)


DSHOW = [(0, ""), (1, "DemoCreator Virtual Camera"), (2, "OBS Virtual Camera")]


def test_list_cameras_names_empty_entries(monkeypatch):
    fake_enum(monkeypatch, DSHOW, [])
    cams = list_cameras()
    assert [(c.index, c.name) for c in cams] == [(0, "Camera 0"), (1, DSHOW[1][1]), (2, DSHOW[2][1])]
    assert all(c.backend == cv2.CAP_DSHOW for c in cams)


def test_list_cameras_borrows_media_foundation_name(monkeypatch):
    fake_enum(monkeypatch, DSHOW, [(0, "USB2.0 HD UVC WebCam"), (1, "OBS Virtual Camera")])
    assert list_cameras()[0].name == "USB2.0 HD UVC WebCam"


def test_list_cameras_ignores_virtual_or_ambiguous_guess(monkeypatch):
    fake_enum(monkeypatch, DSHOW, [(0, "Windows Virtual Camera")])
    assert list_cameras()[0].name == "Camera 0"
    fake_enum(monkeypatch, DSHOW, [(0, "Cam A"), (1, "Cam B")])
    assert list_cameras()[0].name == "Camera 0"


# ---------------------------------------------------------------- capture thread


class FakeCap:
    """Stands in for cv2.VideoCapture; serves 640x360 frames at ~100 fps."""

    fail = threading.Event()
    opened = 0
    sets: ClassVar[list[int]] = []

    def __init__(self, index, api):
        assert api == cv2.CAP_DSHOW
        FakeCap.opened += 1
        self.n = 0

    def isOpened(self):
        return True

    def set(self, prop, value):
        FakeCap.sets.append(prop)
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 360, cv2.CAP_PROP_FPS: 30}.get(prop, 0)

    def read(self):
        time.sleep(0.01)
        if FakeCap.fail.is_set():
            return False, None
        self.n += 1
        return True, np.full((360, 640, 3), self.n % 256, np.uint8)

    def release(self):
        pass


@pytest.fixture
def fake_cap(monkeypatch):
    FakeCap.fail.clear()
    FakeCap.opened = 0
    FakeCap.sets = []
    monkeypatch.setattr(camera.cv2, "VideoCapture", FakeCap)
    monkeypatch.setattr(camera, "pick_camera", lambda preferred="": CameraInfo(0, "Fake Webcam"))
    return FakeCap


def wait_for(cond, timeout=3.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_mjpg_is_set_before_size(fake_cap):
    with Camera() as cam:
        assert cam.wait_latest(2.0) is not None
    sets = fake_cap.sets
    assert sets[0] == cv2.CAP_PROP_FOURCC
    assert sets.index(cv2.CAP_PROP_FOURCC) < sets.index(cv2.CAP_PROP_FRAME_WIDTH) < sets.index(cv2.CAP_PROP_FPS)


def test_frames_resized_newest_first_and_never_repeated(fake_cap):
    with Camera(width=1280, height=720) as cam:
        p1 = cam.wait_latest(2.0)
        assert p1 is not None and p1.frame.shape == (720, 1280, 3) and p1.frame.dtype == np.uint8
        time.sleep(0.1)  # several frames arrive; only the newest is kept
        p2 = cam.wait_latest(1.0)
        assert p2.seq >= p1.seq + 3 and p2.t > p1.t
        p3 = cam.wait_latest(1.0)
        assert p3 is None or p3.seq > p2.seq
        assert cam.connected and cam.error is None
        assert wait_for(lambda: cam.fps > 20, 2.5)
        assert cam.info.name == "Fake Webcam"


def test_disconnect_and_reconnect(fake_cap):
    cam = Camera(retry_s=0.1).start()
    try:
        assert cam.wait_latest(2.0) is not None
        fake_cap.fail.set()
        assert wait_for(lambda: not cam.connected, 2.0)
        assert cam.error and cam.fps == 0.0
        assert cam.wait_latest(0.2) is None
        fake_cap.fail.clear()
        assert wait_for(lambda: cam.connected, 2.0)
        assert fake_cap.opened >= 2
        assert cam.wait_latest(1.0) is not None
    finally:
        cam.stop()
    assert not cam._thread.is_alive()


def test_open_failure_reports_error_and_retries(monkeypatch):
    calls = []

    def no_camera(preferred=""):
        calls.append(1)
        raise CameraError("no physical webcam found; cameras: [2] OBS Virtual Camera")

    monkeypatch.setattr(camera, "pick_camera", no_camera)
    cam = Camera(retry_s=0.05).start()
    try:
        assert wait_for(lambda: len(calls) >= 3, 2.0)
        assert not cam.connected and "no physical webcam" in cam.error
        assert cam.wait_latest(0.05) is None
    finally:
        cam.stop()
