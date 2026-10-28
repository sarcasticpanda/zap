"""DirectShow -> Media Foundation fallback in the capture thread, with fake devices (no webcam)."""

from __future__ import annotations

import time
from typing import ClassVar

import cv2
import numpy as np
import pytest

from zap.devices import camera
from zap.devices.camera import Camera, CameraError, CameraInfo, pick_msmf_camera

MSMF_WEBCAM = CameraInfo(0, "USB2.0 HD UVC WebCam", cv2.CAP_MSMF)


class Cap:
    """VideoCapture fake: DirectShow (or both) can be made to refuse to open or to deliver."""

    opened: ClassVar[list[int]] = []
    dshow_opens = False
    dshow_reads = False
    msmf_ok = True

    def __init__(self, index, backend):
        self.backend = backend
        Cap.opened.append(backend)

    def isOpened(self):
        return Cap.dshow_opens if self.backend == cv2.CAP_DSHOW else Cap.msmf_ok

    def set(self, prop, value):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: 1280, cv2.CAP_PROP_FRAME_HEIGHT: 720, cv2.CAP_PROP_FPS: 30}.get(prop, 0)

    def read(self):
        time.sleep(0.005)
        ok = Cap.dshow_reads if self.backend == cv2.CAP_DSHOW else Cap.msmf_ok
        return (True, np.full((720, 1280, 3), 7, np.uint8)) if ok else (False, None)

    def release(self):
        pass


@pytest.fixture
def devices(monkeypatch):
    Cap.opened, Cap.dshow_opens, Cap.dshow_reads, Cap.msmf_ok = [], False, False, True
    monkeypatch.setattr(camera.cv2, "VideoCapture", Cap)
    monkeypatch.setattr(camera, "pick_camera", lambda preferred="": CameraInfo(0, "Camera 0"))
    monkeypatch.setattr(camera, "list_msmf_cameras", lambda: [MSMF_WEBCAM])
    return Cap


def test_dshow_refuses_to_open_media_foundation_used(devices):
    with Camera() as cam:
        pkt = cam.wait_latest(2.0)
        assert pkt is not None and pkt.frame.shape == (720, 1280, 3)
        assert cam.info == MSMF_WEBCAM and cam.backend_name == "Media Foundation"
    assert devices.opened[:2] == [cv2.CAP_DSHOW, cv2.CAP_MSMF]


def test_dshow_opens_but_delivers_nothing(devices):
    devices.dshow_opens = True
    with Camera() as cam:
        assert cam.wait_latest(2.0) is not None
        assert cam.info.backend == cv2.CAP_MSMF


def test_dshow_used_when_it_works(devices):
    devices.dshow_opens = devices.dshow_reads = True
    with Camera() as cam:
        assert cam.wait_latest(2.0) is not None
        assert cam.backend_name == "DirectShow"
    assert cv2.CAP_MSMF not in devices.opened


def test_backend_setting_restricts_the_search(devices):
    with Camera(backend="msmf") as cam:
        assert cam.wait_latest(2.0) is not None
    assert cv2.CAP_DSHOW not in devices.opened
    devices.opened.clear()
    cam = Camera(backend="dshow", retry_s=0.05).start()
    try:
        time.sleep(0.2)
        assert not cam.connected and "could not open camera" in cam.error
    finally:
        cam.stop()
    assert set(devices.opened) == {cv2.CAP_DSHOW}
    with pytest.raises(CameraError):
        Camera(backend="v4l2")


def test_nothing_opens_reports_both(devices):
    devices.msmf_ok = False
    cam = Camera(retry_s=0.05).start()
    try:
        time.sleep(0.2)
        assert "DirectShow" in cam.error and "Media Foundation" in cam.error
    finally:
        cam.stop()


def test_reconnect_tries_the_working_backend_first(devices):
    cam = Camera(retry_s=0.01)
    assert cam._open() is not None and cam.info.backend == cv2.CAP_MSMF
    devices.opened.clear()
    assert cam._open() is not None
    assert devices.opened == [cv2.CAP_MSMF]


def test_pick_msmf_refuses_own_output_and_prefers_by_name(monkeypatch):
    cams = [CameraInfo(0, "OBS Virtual Camera", cv2.CAP_MSMF), CameraInfo(1, "Some Virtual Cam", cv2.CAP_MSMF),
            CameraInfo(2, "Integrated Camera", cv2.CAP_MSMF), CameraInfo(3, "Logitech BRIO", cv2.CAP_MSMF)]
    monkeypatch.setattr(camera, "list_msmf_cameras", lambda: cams)
    assert pick_msmf_camera("").name == "Integrated Camera"
    assert pick_msmf_camera("brio").name == "Logitech BRIO"
    assert pick_msmf_camera("obs").name == "Integrated Camera"  # never ZAP's own output
    monkeypatch.setattr(camera, "list_msmf_cameras", lambda: cams[:2])
    assert pick_msmf_camera("") is None


def test_pick_msmf_survives_enumeration_errors(monkeypatch):
    def boom():
        raise OSError("no Media Foundation")

    monkeypatch.setattr(camera, "list_msmf_cameras", boom)
    assert pick_msmf_camera("") is None


def test_camera_info_names_its_backend():
    assert str(MSMF_WEBCAM) == "0: USB2.0 HD UVC WebCam (Media Foundation)"
    assert CameraInfo(1, "X").backend_name == "DirectShow"
