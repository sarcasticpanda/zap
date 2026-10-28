"""Microphone capture for snap detection.

The stream opens the mic in WASAPI *shared* mode so Zoom keeps using it at the
same time. The PortAudio callback runs on a real-time thread, so it only
copies the block and its capture time into a queue; a separate analysis thread
batches queued blocks (~20 ms at a time, because scipy's filters cost the same
for 128 or 1000 samples) and runs the detector.

Nothing is ever written to disk. Any device error is reported through
``error`` and the stream shuts itself down; the app then falls back to
vision-only snaps.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

from zap.gestures.snap_audio import AudioCandidate, SnapAudioDetector

log = logging.getLogger(__name__)

POLL_S = 0.02  # analysis thread batching interval
STALL_S = 1.5  # no audio for this long = the device died
LEVEL_TAU_S = 0.1  # time constant of the HUD level meter
CONTIGUOUS_TOL_S = 0.004  # blocks whose timestamps line up within this are batched


def find_input_device(name: str | None = None) -> int:
    """PortAudio index of the microphone to use (WASAPI when available).

    ``name`` matches as a case-insensitive substring; when nothing matches the
    system default microphone is used. Raises LookupError when there is no
    input device at all.
    """
    apis = sd.query_hostapis()
    devices = sd.query_devices()
    wasapi = next((i for i, a in enumerate(apis) if "WASAPI" in a["name"]), None)
    inputs = [
        i for i, d in enumerate(devices) if d["max_input_channels"] > 0 and (wasapi is None or d["hostapi"] == wasapi)
    ]
    if name:
        match = [i for i in inputs if name.lower() in devices[i]["name"].lower()]
        if match:
            return match[0]
        log.warning("microphone %r not found, using the default microphone", name)
    default = apis[wasapi]["default_input_device"] if wasapi is not None else sd.default.device[0]
    if default is None or default < 0:
        if not inputs:
            raise LookupError("no microphone found")
        return inputs[0]
    return int(default)


class MicStream:
    """Mic -> SnapAudioDetector on a background thread.

    ``start()`` returns False (and sets ``error``) if the device cannot be
    opened. Call ``drain()`` once per frame to collect judged candidates.
    ``level`` and ``noise_floor`` (dBFS) feed the HUD meter.
    """

    def __init__(self, detector: SnapAudioDetector, device: str | None = None, blocksize: int = 128) -> None:
        self.detector = detector
        self.device = device
        self.rate = detector.rate
        self.blocksize = blocksize
        self.error: str | None = None
        self.device_name = ""
        self.overflows = 0
        self._blocks: deque[tuple[np.ndarray, float]] = deque(maxlen=int(4.0 * self.rate / blocksize))
        self._candidates: deque[AudioCandidate] = deque(maxlen=256)
        self._stream: sd.InputStream | None = None
        self._thread: threading.Thread | None = None
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._latency = 0.0
        self._ms = 0.0  # smoothed mean square for the level meter
        self._last_audio = 0.0
        self._finished = False

    # ------------------------------------------------------------ public
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def level(self) -> float:
        """Recent RMS level in dBFS (-120 when silent)."""
        return 10.0 * math.log10(max(self._ms, 1e-12))

    @property
    def noise_floor(self) -> float:
        """Detector HF noise floor in dB (mean square, relative to full scale)."""
        return self.detector.noise_floor_dbfs

    @property
    def pending_onset(self) -> float | None:
        return self.detector.pending_onset

    def gate(self, until: float) -> None:
        """Ignore snaps until ``until`` (perf_counter), e.g. while ZAP plays a sound."""
        self.detector.gate(until)

    def drain(self) -> list[AudioCandidate]:
        out = []
        while self._candidates:
            out.append(self._candidates.popleft())
        return out

    def start(self) -> bool:
        if self.running:
            return True
        self.error = None
        self._halt.clear()
        self._finished = False
        try:
            self._stream = self._open()
            self._stream.start()
        except (sd.PortAudioError, LookupError, OSError, ValueError, RuntimeError) as e:
            self._close_stream()
            self.error = f"microphone unavailable: {e}"
            log.warning(self.error)
            return False
        self._latency = float(self._stream.latency)
        self._last_audio = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="zap-snap-audio", daemon=True)
        self._thread.start()
        log.info("microphone %r open (%d Hz, latency %.1f ms)", self.device_name, self.rate, self._latency * 1000)
        return True

    def stop(self) -> None:
        self._halt.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._thread = None
        self._close_stream()

    # ------------------------------------------------------------ device
    def _open(self) -> sd.InputStream:
        index = find_input_device(self.device)
        info = sd.query_devices(index)
        self.device_name = info["name"]
        wasapi = "WASAPI" in sd.query_hostapis(info["hostapi"])["name"]
        extra = sd.WasapiSettings(exclusive=False, auto_convert=True) if wasapi else None
        error: Exception | None = None
        # Mono first; some drivers only open with their native channel count.
        for channels in dict.fromkeys((1, min(2, int(info["max_input_channels"])))):
            try:
                return sd.InputStream(
                    samplerate=self.rate,
                    blocksize=self.blocksize,
                    device=index,
                    channels=channels,
                    dtype="float32",
                    latency="low",
                    extra_settings=extra,
                    callback=self._callback,
                    finished_callback=self._on_finished,
                )
            except sd.PortAudioError as e:
                error = e
        raise error or RuntimeError("could not open the microphone")

    def _close_stream(self) -> None:
        with self._lock:
            s, self._stream = self._stream, None
        if s is not None:
            try:
                s.abort()
                s.close()
            except Exception:
                log.debug("closing the microphone failed", exc_info=True)

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        # Real-time thread: copy and timestamp only.
        t_cb = time.perf_counter()
        self._blocks.append((indata[:, 0].copy(), self.block_start(t_cb, frames, time_info)))
        if status.input_overflow:
            self.overflows += 1

    def block_start(self, t_cb: float, frames: int, time_info) -> float:
        """perf_counter time of the block's first sample."""
        adc = time_info.inputBufferAdcTime
        if adc:
            now = time_info.currentTime or (self._stream.time if self._stream is not None else 0.0)
            age = now - adc
            if 0.0 <= age < 1.0:
                return t_cb - age - frames / self.rate
        return t_cb - frames / self.rate - self._latency

    def _on_finished(self) -> None:
        self._finished = True

    # ------------------------------------------------------------ analysis
    def _run(self) -> None:
        try:
            while not self._halt.wait(POLL_S):
                if self._blocks:
                    self._last_audio = time.perf_counter()
                    for block, t0 in self._batches():
                        self._analyse(block, t0)
                elif self._finished or time.perf_counter() - self._last_audio > STALL_S:
                    self._fail("microphone stopped delivering audio")
                    return
        except Exception as e:
            log.exception("snap audio analysis crashed")
            self._fail(f"snap audio analysis crashed: {e}")

    def _batches(self) -> list[tuple[np.ndarray, float]]:
        """Pop queued blocks, joining runs whose timestamps are contiguous."""
        runs: list[tuple[list[np.ndarray], float, int]] = []
        while self._blocks:
            block, t0 = self._blocks.popleft()
            if runs:
                parts, start, n = runs[-1]
                if abs(start + n / self.rate - t0) < CONTIGUOUS_TOL_S:
                    parts.append(block)
                    runs[-1] = (parts, start, n + block.size)
                    continue
            runs.append(([block], t0, block.size))
        return [(np.concatenate(parts) if len(parts) > 1 else parts[0], start) for parts, start, _ in runs]

    def _analyse(self, block: np.ndarray, t0: float) -> None:
        ms = float(np.dot(block, block)) / max(1, block.size)
        a = 1.0 - math.exp(-block.size / self.rate / LEVEL_TAU_S)
        self._ms += a * (ms - self._ms)
        for c in self.detector.process(block, t0):
            self._candidates.append(c)

    def _fail(self, message: str) -> None:
        self.error = message
        log.warning("%s; snaps fall back to vision only", message)
        self._halt.set()
        self._close_stream()
