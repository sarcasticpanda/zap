"""Webcam capture on a background thread.

Picking the camera is the risky part: Windows lists virtual cameras (OBS, the
one ZAP itself feeds, DemoCreator, ...) next to the real webcam, and some
webcams report an empty DirectShow name. Capturing from "OBS Virtual Camera"
would feed ZAP its own output, so names containing "obs" or "zap" are refused
even when the user asks for them.

Some webcams are listed by DirectShow (with an empty name) but refuse to open
there, while Media Foundation opens them fine. So the chosen DirectShow camera
is tried first and, if it will not deliver a frame, the first physical Media
Foundation camera is used instead. Media Foundation does not list
DirectShow-only virtual cameras (OBS, DemoCreator), so the fallback cannot
capture ZAP's own output; the name checks apply there too.

The capture thread blocks in ``VideoCapture.read()`` (OpenCV releases the GIL
there) and keeps only the newest frame, so a slow consumer always gets the
freshest picture instead of a growing backlog. If the camera disappears it
reconnects every couple of seconds, re-resolving the device because DirectShow
indexes shift when devices come and go.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import cv2
import numpy as np

from zap.render.blend import cover_fit
from zap.signals import FramePacket

if TYPE_CHECKING:
    from zap.config import Config

log = logging.getLogger(__name__)

# Substrings (case-insensitive) of known virtual cameras. They are skipped when
# picking a camera automatically; phone-as-webcam apps (Camo) are listed too, so
# a laptop's own webcam wins unless the user names the virtual one in config.
VIRTUAL_HINTS = (
    "obs",
    "virtual",
    "zap camera",
    "unity video capture",
    "xsplit",
    "manycam",
    "snap camera",
    "democreator",
    "vcam",
    "softcam",
    "camo",
)
# Never capture these, even on request: they would capture ZAP's own output.
SELF_CAPTURE_HINTS = ("obs", "zap")

BACKEND_NAMES = {cv2.CAP_DSHOW: "DirectShow", cv2.CAP_MSMF: "Media Foundation"}
BACKENDS = {"auto": None, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}  # config camera.backend

RETRY_S = 2.0
STALL_S = 2.0  # no frame for this long counts as disconnected
FPS_WINDOW_S = 1.0
_INDEX_RE = re.compile(r"^\s*(?:camera\s*)?#?(\d+)\s*$", re.IGNORECASE)


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraInfo:
    index: int  # DirectShow device index, as cv2.VideoCapture(index, cv2.CAP_DSHOW) expects
    name: str
    backend: int = cv2.CAP_DSHOW

    @property
    def virtual(self) -> bool:
        return is_virtual(self.name)

    @property
    def backend_name(self) -> str:
        return BACKEND_NAMES.get(self.backend, str(self.backend))

    def __str__(self) -> str:
        return f"{self.index}: {self.name} ({self.backend_name})"


def is_virtual(name: str) -> bool:
    n = name.lower()
    return any(h in n for h in VIRTUAL_HINTS)


def is_self_capture(name: str) -> bool:
    n = name.lower()
    return any(h in n for h in SELF_CAPTURE_HINTS)


def _msmf_names() -> list[str]:
    from cv2_enumerate_cameras import enumerate_cameras

    return [c.name for c in enumerate_cameras(cv2.CAP_MSMF) if c.name]


def list_cameras() -> list[CameraInfo]:
    """DirectShow cameras in index order. An empty name becomes "Camera <index>", unless
    exactly one camera is unnamed and Media Foundation knows exactly one name that
    DirectShow is missing, in which case that name is used."""
    from cv2_enumerate_cameras import enumerate_cameras

    raw = [(int(c.index), (c.name or "").strip()) for c in enumerate_cameras(cv2.CAP_DSHOW)]
    unnamed = [i for i, n in raw if not n]
    guess = None
    if len(unnamed) == 1:
        try:
            known = {n for _, n in raw}
            extra = [n for n in _msmf_names() if n not in known]
            # a virtual-looking guess would make the real webcam look virtual and get skipped
            guess = extra[0] if len(extra) == 1 and not is_virtual(extra[0]) else None
        except Exception:  # best effort only
            log.debug("Media Foundation camera names unavailable", exc_info=True)
    return [CameraInfo(i, n or guess or f"Camera {i}", cv2.CAP_DSHOW) for i, n in raw]


def list_msmf_cameras() -> list[CameraInfo]:
    """Media Foundation cameras (real names; DirectShow-only virtual cameras are not listed)."""
    from cv2_enumerate_cameras import enumerate_cameras

    return [
        CameraInfo(int(c.index), (c.name or "").strip() or f"Camera {c.index}", cv2.CAP_MSMF)
        for c in enumerate_cameras(cv2.CAP_MSMF)
    ]


def pick_msmf_camera(preferred: str = "") -> CameraInfo | None:
    """The Media Foundation fallback: the preferred camera by name if listed, else the first
    physical one; never one that looks like ZAP's own output. None when there is none."""
    try:
        cams = [c for c in list_msmf_cameras() if not is_self_capture(c.name)]
    except Exception:  # the fallback is best effort
        log.debug("Media Foundation cameras unavailable", exc_info=True)
        return None
    pref = preferred.strip().lower()
    if pref and not _INDEX_RE.match(pref):
        named = [c for c in cams if pref in c.name.lower()]
        if named:
            return named[0]
    return next((c for c in cams if not c.virtual), None)


def _describe(cameras: list[CameraInfo]) -> str:
    if not cameras:
        return "no cameras found"
    return "cameras: " + ", ".join(f"[{c.index}] {c.name}" for c in cameras)


def pick_camera(preferred: str = "", cameras: list[CameraInfo] | None = None) -> CameraInfo:
    """Choose the capture camera.

    `preferred` (config ``camera.name``) matches a camera name case-insensitively
    by substring, or an index written as "1", "#1" or "Camera 1". Cameras named
    like OBS/ZAP are refused either way. With no preference, or when the preferred
    camera is missing, the first non-virtual camera is used.
    """
    cams = list_cameras() if cameras is None else cameras
    pref = preferred.strip()
    if pref:
        m = _INDEX_RE.match(pref)
        if m:
            hits = [c for c in cams if c.index == int(m.group(1))]
        else:
            hits = [c for c in cams if pref.lower() in c.name.lower()]
        usable = [c for c in hits if not is_self_capture(c.name)]
        if usable:
            return usable[0]
        if hits:
            raise CameraError(
                f"camera.name = '{preferred}' matches '{hits[0].name}', which is ZAP's own output "
                f"(capturing it would loop). Use your physical webcam; {_describe(cams)}"
            )
        log.warning("camera '%s' not found (%s); picking one automatically", preferred, _describe(cams))
    for c in cams:
        if not is_virtual(c.name):
            return c
    raise CameraError(
        "no physical webcam found; "
        + _describe(cams)
        + ". Plug in a webcam, close apps that hide it, or set camera.name in config.toml."
    )


class Camera:
    """Newest-frame-wins capture thread.

    ``wait_latest(timeout)`` returns a :class:`FramePacket` newer than the last one
    it returned (or None on timeout). Frames are BGR uint8 at ``out_size``.
    """

    def __init__(
        self,
        preferred: str = "",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        out_size: tuple[int, int] | None = None,
        retry_s: float = RETRY_S,
        backend: str = "auto",
        mirror: bool = False,
    ) -> None:
        if backend not in BACKENDS:
            raise CameraError(f"camera.backend must be one of {', '.join(BACKENDS)}, got '{backend}'")
        self.preferred = preferred
        self.backend = backend
        self.width, self.height, self.req_fps = width, height, fps
        self.out_w, self.out_h = out_size or (width, height)
        self.retry_s = retry_s
        self.mirror = mirror  # flip like a selfie camera before anything else sees the frame
        self.info: CameraInfo | None = None
        self.actual: tuple[int, int, float] | None = None  # what the driver agreed to (w, h, fps)
        self._cond = threading.Condition()
        self._packet: FramePacket | None = None
        self._returned_seq = -1
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._open_flag = False
        self._last_t = 0.0
        self._fps = 0.0
        self._error: str | None = None
        self._first: np.ndarray | None = None  # the frame read while probing the device
        self._prefer_backend: int | None = None  # the backend that worked last time

    @classmethod
    def from_config(cls, cfg: Config) -> Camera:
        c = cfg.camera
        return cls(c.name, c.width, c.height, c.fps, out_size=(cfg.output.width, cfg.output.height),
                   backend=c.backend, mirror=c.mirror)

    @property
    def backend_name(self) -> str:
        """The capture backend in use ("DirectShow" / "Media Foundation"), or "" before opening."""
        return self.info.backend_name if self.info is not None else ""

    # ------------------------------------------------------------ public API
    def start(self) -> Camera:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="zap-camera", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self.join(timeout)

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def wait_latest(self, timeout: float = 1.0) -> FramePacket | None:
        deadline = time.perf_counter() + timeout
        with self._cond:
            while self._packet is None or self._packet.seq <= self._returned_seq:
                left = deadline - time.perf_counter()
                if left <= 0 or self._stop.is_set():
                    return None
                self._cond.wait(left)
            self._returned_seq = self._packet.seq
            return self._packet

    @property
    def fps(self) -> float:
        """Frames per second actually delivered, measured over about a second."""
        return self._fps if self.connected else 0.0

    @property
    def connected(self) -> bool:
        return self._open_flag and time.perf_counter() - self._last_t < STALL_S

    @property
    def error(self) -> str | None:
        if self._open_flag and not self.connected and self._last_t > 0:
            return "camera stopped sending frames"
        return self._error

    # ------------------------------------------------------------ thread
    def _candidates(self) -> list[CameraInfo]:
        """Cameras to try, in order: the DirectShow pick, then the Media Foundation fallback."""
        want = BACKENDS[self.backend]
        out: list[CameraInfo] = []
        errors: list[str] = []
        if want in (None, cv2.CAP_DSHOW):
            try:
                out.append(pick_camera(self.preferred))
            except CameraError as e:
                errors.append(str(e))
            except Exception as e:  # noqa: BLE001 - enumeration itself failed
                errors.append(f"could not list cameras: {e}")
        if want in (None, cv2.CAP_MSMF):
            msmf = pick_msmf_camera(self.preferred)
            if msmf is not None:
                out.append(msmf)
        if not out:
            self._error = errors[0] if errors else "no physical webcam found"
        elif self._prefer_backend is not None:  # reconnecting: the backend that worked last time first
            out.sort(key=lambda c: c.backend != self._prefer_backend)
        return out

    def _open(self) -> cv2.VideoCapture | None:
        cams = self._candidates()
        for k, info in enumerate(cams):
            cap = self._open_one(info)
            if cap is not None:
                if k:
                    log.info("camera: %s did not deliver frames; using %s instead", cams[0], info)
                return cap
        if cams:
            names = " or ".join(f"'{c.name}' ({c.backend_name})" for c in cams)
            self._error = f"could not open camera {names} (is another app using it?)"
        return None

    def _open_one(self, info: CameraInfo) -> cv2.VideoCapture | None:
        """Open one device as MJPG at the requested size and read a frame; None if it won't deliver."""
        cap = cv2.VideoCapture(info.index, info.backend)
        if not cap.isOpened():
            cap.release()
            return None
        # DirectShow only honours MJPG if it is set before the size; MJPG is what makes 720p30 possible
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.req_fps)
        ok, frame = cap.read()
        if not ok or frame is None:
            log.info("camera %s opened but delivered no frame", info)
            cap.release()
            return None
        self._first = frame
        self.info = info
        self._prefer_backend = info.backend
        self.actual = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(cap.get(cv2.CAP_PROP_FPS)),
        )
        log.info("camera '%s' (index %d, %s) opened at %dx%d @ %.0f fps",
                 info.name, info.index, info.backend_name, *self.actual)
        return cap

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        if self.mirror:
            frame = cv2.flip(frame, 1)  # a new contiguous array
        h, w = frame.shape[:2]
        if (w, h) == (self.out_w, self.out_h):
            return frame
        if abs(w / h - self.out_w / self.out_h) > 0.01:
            return cover_fit(frame, self.out_w, self.out_h)
        interp = cv2.INTER_AREA if w > self.out_w else cv2.INTER_LINEAR
        return cv2.resize(frame, (self.out_w, self.out_h), interpolation=interp)

    def _run(self) -> None:
        cap: cv2.VideoCapture | None = None
        fails = 0
        win_t0, win_n = 0.0, 0
        try:
            while not self._stop.is_set():
                if cap is None:
                    cap = self._open()
                    if cap is None:
                        log.warning("%s; retrying in %.0f s", self._error, self.retry_s)
                        self._stop.wait(self.retry_s)
                        continue
                    fails, win_t0, win_n = 0, time.perf_counter(), 0
                    self._last_t = time.perf_counter()
                    self._open_flag = True
                if self._first is not None:  # the probe frame from _open_one
                    ok, frame, self._first = True, self._first, None
                else:
                    ok, frame = cap.read()
                now = time.perf_counter()
                if not ok or frame is None:
                    fails += 1
                    if fails >= 5 or now - self._last_t > STALL_S:
                        self._error = "camera disconnected"
                        log.warning("camera disconnected; reconnecting every %.0f s", self.retry_s)
                        self._open_flag = False
                        cap.release()
                        cap = None
                        self._stop.wait(self.retry_s)
                    else:
                        self._stop.wait(0.05)
                    continue
                fails = 0
                self._error = None
                packet = FramePacket(self._fit(frame), now, self._seq)
                self._seq += 1
                self._last_t = now
                win_n += 1
                if now - win_t0 >= FPS_WINDOW_S:
                    self._fps = win_n / (now - win_t0)
                    win_t0, win_n = now, 0
                with self._cond:
                    self._packet = packet
                    self._cond.notify_all()
        except Exception as e:  # keep the app alive; report through .error
            self._error = f"camera thread crashed: {e}"
            log.exception("camera thread crashed")
        finally:
            self._open_flag = False
            if cap is not None:
                cap.release()
