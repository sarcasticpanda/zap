"""Shared test safety nets: no test may open a real webcam."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_real_media_foundation_cameras(monkeypatch):
    """Camera's Media Foundation fallback would find (and open) the real webcam; tests that
    exercise it replace this with a fake list."""
    from zap.devices import camera

    monkeypatch.setattr(camera, "list_msmf_cameras", list)
