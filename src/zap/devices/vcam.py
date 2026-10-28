"""Output to Zoom through the OBS Virtual Camera.

pyvirtualcam's OBS backend writes into the shared-memory queue that OBS
Studio's DirectShow filter reads, so Zoom sees "OBS Virtual Camera" without OBS
running. Its errors are terse and technical; they are translated here into
messages a user can act on. Pacing is left to the webcam (one output frame per
captured frame), so ``sleep_until_next_frame`` is deliberately not used.
"""

from __future__ import annotations

import logging
from typing import Self

import numpy as np

log = logging.getLogger(__name__)

NOT_INSTALLED = "OBS Virtual Camera is not installed. Install OBS Studio (it registers the camera) and try again."
BUSY = "OBS Virtual Camera is busy: stop 'Start Virtual Camera' in OBS, or close the other ZAP window."


class VirtualCamError(RuntimeError):
    pass


def friendly_error(err: BaseException) -> str:
    """Map pyvirtualcam's RuntimeError text to an actionable message."""
    msg = str(err)
    low = msg.lower()
    if "device not found" in low or "not installed" in low:
        return NOT_INSTALLED
    if "could not be started" in low or "in use" in low:
        return BUSY
    return f"Could not open the OBS Virtual Camera: {msg}"


class VirtualCam:
    """BGR frames in, OBS Virtual Camera out."""

    def __init__(self, width: int = 1280, height: int = 720, fps: float = 30) -> None:
        import pyvirtualcam

        self.width, self.height, self.fps = int(width), int(height), float(fps)
        self.shape = (self.height, self.width, 3)
        try:
            self._cam = pyvirtualcam.Camera(
                self.width, self.height, self.fps, fmt=pyvirtualcam.PixelFormat.BGR, backend="obs"
            )
        except RuntimeError as e:
            raise VirtualCamError(friendly_error(e)) from e
        log.info("virtual camera '%s' open at %dx%d @ %.0f fps", self._cam.device, self.width, self.height, self.fps)

    @property
    def device(self) -> str:
        return self._cam.device if self._cam is not None else ""

    @property
    def frames_sent(self) -> int:
        return self._cam.frames_sent if self._cam is not None else 0

    def send(self, frame: np.ndarray) -> None:
        if self._cam is None:
            raise VirtualCamError("virtual camera is closed")
        if frame.shape != self.shape:
            raise ValueError(f"frame shape {frame.shape} != {self.shape}")
        if frame.dtype != np.uint8 or not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
        self._cam.send(frame)

    def close(self) -> None:
        cam, self._cam = self._cam, None
        if cam is not None:
            try:
                cam.close()
            except Exception:
                log.exception("error closing the virtual camera")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
