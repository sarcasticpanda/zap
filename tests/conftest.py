"""Shared test safety nets: no test may open a real webcam, and every test sees the same OS paths."""

from __future__ import annotations

import os
import sys

import pytest

if sys.platform == "linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("PYNPUT_BACKEND", "dummy")  # pynput's X11 backend needs a display; tests press no keys


@pytest.fixture(autouse=True)
def _no_real_media_foundation_cameras(monkeypatch):
    """Camera's Media Foundation fallback would find (and open) the real webcam; tests that
    exercise it replace this with a fake list. The camera tests describe Windows' DirectShow then
    Media Foundation order on every OS; tests of the macOS and Linux paths set PLATFORM themselves."""
    from zap.devices import camera

    monkeypatch.setattr(camera, "list_msmf_cameras", list)
    monkeypatch.setattr(camera, "PLATFORM", "win32")
