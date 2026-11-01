"""Output to your call app through a virtual camera.

Windows: pyvirtualcam's OBS backend writes into the shared-memory queue that OBS
Studio's DirectShow filter reads, so call apps see "OBS Virtual Camera" without OBS
running. macOS: the same backend feeds OBS 30+'s camera extension. Linux: a
v4l2loopback device, preferably one labelled "ZAP Camera". pyvirtualcam's errors
are terse and technical; they are translated here into messages a user can act
on. Pacing is left to the webcam (one output frame per captured frame), so
``sleep_until_next_frame`` is deliberately not used.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Self

import numpy as np

log = logging.getLogger(__name__)

PLATFORM = sys.platform
LINUX_LABEL = "ZAP Camera"
NOT_INSTALLED = "OBS Virtual Camera is not installed. Install OBS Studio (it registers the camera) and try again."
NOT_INSTALLED_MAC = (
    "OBS Virtual Camera isn't set up. Install OBS Studio 30 or newer, press 'Start Virtual Camera' in it once and "
    "allow its camera extension in System Settings, then quit OBS and start ZAP again."
)
MODPROBE = f'sudo modprobe v4l2loopback devices=1 exclusive_caps=1 card_label="{LINUX_LABEL}"'
NOT_INSTALLED_LINUX = (
    f"No virtual camera device. Load one with: {MODPROBE} (install the v4l2loopback-dkms package first if "
    "modprobe can't find it)."
)
NO_ACCESS_LINUX = "No permission to use the virtual camera: run 'sudo usermod -aG video $USER', then log in again."
BUSY = "OBS Virtual Camera is busy: stop 'Start Virtual Camera' in OBS, or close the other ZAP window."


class VirtualCamError(RuntimeError):
    pass


def friendly_error(err: BaseException) -> str:
    """Map pyvirtualcam's error text to an actionable message."""
    msg = str(err)
    low = msg.lower()
    if PLATFORM == "linux":
        if "permission" in low or "denied" in low:
            return NO_ACCESS_LINUX
        if "loopback" in low or "no such" in low or "not found" in low:
            return NOT_INSTALLED_LINUX
        return f"Could not open the virtual camera: {msg}"
    if "device not found" in low or "not installed" in low or "couldn't start stream" in low:
        return NOT_INSTALLED_MAC if PLATFORM == "darwin" else NOT_INSTALLED
    if "could not be started" in low or "in use" in low:
        return BUSY
    return f"Could not open the OBS Virtual Camera: {msg}"


def camera_name(platform: str = PLATFORM) -> str:
    """What call apps call ZAP's camera on this OS."""
    return LINUX_LABEL if platform == "linux" else "OBS Virtual Camera"


def linux_device(label: str = LINUX_LABEL, sysfs: Path = Path("/sys/class/video4linux")) -> str | None:
    """/dev/videoN of the loaded v4l2loopback device with this label, if any."""
    for d in sorted(sysfs.glob("video*")):
        try:
            if (d / "name").read_text(encoding="utf-8").strip() == label:
                return f"/dev/{d.name}"
        except OSError:
            log.debug("can't read %s", d, exc_info=True)
    return None


class VirtualCam:
    """BGR frames in, virtual camera out."""

    def __init__(self, width: int = 1280, height: int = 720, fps: float = 30) -> None:
        import pyvirtualcam

        self.width, self.height, self.fps = int(width), int(height), float(fps)
        self.shape = (self.height, self.width, 3)
        where: dict[str, str] = {"backend": "obs"}
        if PLATFORM == "linux":  # the device labelled ZAP Camera, else the first free loopback device
            where = {"backend": "v4l2loopback"}
            dev = linux_device()
            if dev is not None:
                where["device"] = dev
        try:
            self._cam = pyvirtualcam.Camera(self.width, self.height, self.fps, fmt=pyvirtualcam.PixelFormat.BGR,
                                            **where)
        except (RuntimeError, KeyError, ValueError, OSError) as e:
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
