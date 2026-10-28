"""Sound effects output.

Two modes:

* ``speakers``: SFX go to the default output (WASAPI preferred); only you hear them.
* ``meeting``: SFX are mixed with your microphone into VB-CABLE's "CABLE Input",
  and Zoom uses "CABLE Output" as its microphone, so the meeting hears them.
  Without VB-CABLE this falls back to speakers and says why (``.warning``).

Voices are mixed in the PortAudio callback by :func:`mix_voices`, a pure
function so it can be tested without audio hardware. ``busy_until`` tells the
microphone snap detector when our own sounds might still be echoing back into
the mic, so it can ignore them. An audio problem must never take the video
down, so every device call here is guarded and logged instead of raised.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Self

import numpy as np

try:  # PortAudio missing must not break importing the app
    import sounddevice as sd
except (ImportError, OSError) as _e:  # pragma: no cover - depends on the machine
    sd = None
    _IMPORT_ERROR = str(_e)
else:
    _IMPORT_ERROR = ""

if TYPE_CHECKING:
    from zap.assets import AssetResolver

log = logging.getLogger(__name__)

SAMPLE_RATE = 48000
CHANNELS = 2
LATENCY_S = 0.06  # generous: the callback competes for the GIL with MediaPipe
GATE_TAIL_S = 0.15  # room echo margin added to busy_until
MAX_VOICES = 8
WATCHDOG_S = 2.0
CABLE_NAME = "CABLE Input"
UNGATED_KEYS = ("charge",)  # soft, low-passed pre-roll; must not mute snaps 2 and 3 of a burst


@dataclass(frozen=True)
class Voice:
    data: np.ndarray  # mono float32 at 48 kHz
    pos: int = 0
    gain: float = 1.0
    key: str = ""


def fallback_keys(key: str) -> tuple[str, ...]:
    """Sound keys to try in order: "cutaway_jim" -> ("cutaway_jim", "cutaway")."""
    family = key.split("_", 1)[0]
    return (key, family) if key.startswith("cutaway_") and family != key else (key,)


def mix_voices(out_block: np.ndarray, voices: Sequence[Voice], volume: float) -> list[Voice]:
    """Add `voices` (scaled by `volume`) into `out_block` (frames, channels) float32,
    soft-clip it in place with tanh, and return the voices that still have samples left."""
    n = out_block.shape[0]
    remaining: list[Voice] = []
    if voices:
        acc = np.zeros(n, np.float32)
        for v in voices:  # a handful of voices; each is one vectorized slice add
            seg = v.data[v.pos : v.pos + n]
            if seg.size:
                acc[: seg.size] += seg * np.float32(v.gain)
            if v.pos + n < v.data.size:
                remaining.append(replace(v, pos=v.pos + n))
        acc *= np.float32(volume)
        out_block += acc[:, None]
    np.tanh(out_block, out=out_block)
    return remaining


# ---------------------------------------------------------------- device lookup
def _hostapi_index(name_part: str) -> int | None:
    for i, api in enumerate(sd.query_hostapis()):
        if name_part.lower() in api["name"].lower():
            return i
    return None


def _wasapi_default(kind: str) -> int | None:
    """Default input/output device index of the WASAPI host API, or None."""
    i = _hostapi_index("WASAPI")
    if i is None:
        return None
    dev = sd.query_hostapis(i)[f"default_{kind}_device"]
    return dev if dev is not None and dev >= 0 else None


def find_output_device(name_part: str) -> int | None:
    """Output device whose name contains `name_part` (case-insensitive), WASAPI preferred."""
    wasapi = _hostapi_index("WASAPI")
    hits = [
        (d["hostapi"] != wasapi, i)
        for i, d in enumerate(sd.query_devices())
        if d["max_output_channels"] > 0 and name_part.lower() in d["name"].lower()
    ]
    return min(hits)[1] if hits else None


def _extra_settings(*devices: int | None):
    """Let WASAPI resample when a device's mix format is not 48 kHz."""
    wasapi = _hostapi_index("WASAPI")
    if wasapi is None or not hasattr(sd, "WasapiSettings"):
        return None
    if all(d is not None and sd.query_devices(d)["hostapi"] == wasapi for d in devices):
        settings = sd.WasapiSettings(auto_convert=True)
        return settings if len(devices) == 1 else (settings, settings)
    return None


class SfxPlayer:
    """SfxSink that plays meme sounds; see the module docstring for the modes.

    play() never opens devices or blocks: a small watchdog thread reopens the
    stream in the background when a device disappears.
    """

    def __init__(
        self,
        assets: AssetResolver | None,
        synth: Callable[[str], np.ndarray] | None,
        enabled: bool = True,
        output: str = "speakers",
        volume: float = 0.6,
        ungated_keys: Iterable[str] = UNGATED_KEYS,
    ) -> None:
        self.assets = assets
        self.synth = synth
        self.requested = output if output in ("speakers", "meeting") else "speakers"
        self.mode = self.requested  # what is actually in use: "speakers", "meeting" or "none"
        self.warning = ""
        self.volume = float(volume)
        self._enabled = bool(enabled)
        self._ungated = frozenset(ungated_keys)
        self._lock = threading.Lock()
        self._open_lock = threading.Lock()
        self._pending: list[Voice] = []
        self._voices: list[Voice] = []
        self._clear = False
        self._busy_until = 0.0
        self._stream = None
        self._latency = LATENCY_S
        self._cb_error_logged = False
        self._stop = threading.Event()
        self._watchdog: threading.Thread | None = None
        if sd is None:
            self.mode = "none"
            self.warning = f"no audio: sounddevice unavailable ({_IMPORT_ERROR})"
            log.warning(self.warning)
            return
        if self._wanted():
            self._open()
        self._watchdog = threading.Thread(target=self._watch, name="zap-audio-watchdog", daemon=True)
        self._watchdog.start()

    # ------------------------------------------------------------ public API
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def busy_until(self) -> float:
        """perf_counter time until which our own sounds may still be heard (plus a margin)."""
        return self._busy_until

    @property
    def playing(self) -> bool:
        return time.perf_counter() < self._busy_until

    def play(self, key: str) -> None:
        if not self._enabled or not self._stream_ok():
            return
        try:
            data = self._resolve(key)
            if data is None or data.size == 0:
                return
            with self._lock:
                self._pending.append(Voice(data, 0, 1.0, key))
                excess = len(self._pending) + len(self._voices) - MAX_VOICES
                if excess > 0:
                    del self._pending[:excess]
            if key not in self._ungated:
                end = time.perf_counter() + self._latency + data.size / SAMPLE_RATE + GATE_TAIL_S
                self._busy_until = max(self._busy_until, end)
        except Exception:  # an SFX must never break a frame
            log.exception("could not play sound '%s'", key)

    def stop_all(self) -> None:
        with self._lock:
            self._pending.clear()
            self._clear = True
        self._busy_until = min(self._busy_until, time.perf_counter() + GATE_TAIL_S)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        if not self._enabled:
            self.stop_all()
        elif sd is not None and not self._stream_ok():
            self._open()

    def set_volume(self, volume: float) -> None:
        self.volume = float(min(max(volume, 0.0), 2.0))

    def preload(self, keys: Iterable[str]) -> None:
        """Resolve sounds ahead of time so the first play() does not stall a frame."""
        for k in keys:
            try:
                self._resolve(k)
            except Exception:
                log.exception("could not preload sound '%s'", k)

    def close(self) -> None:
        self._stop.set()
        if self._watchdog is not None and self._watchdog is not threading.current_thread():
            self._watchdog.join(timeout=3.0)
        self._close_stream()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ internals
    def _wanted(self) -> bool:
        # meeting mode carries the microphone to Zoom, so it runs even with SFX muted
        return self._enabled or self.requested == "meeting"

    def _resolve(self, key: str) -> np.ndarray | None:
        """User WAV for the key, then for its family ("cutaway_jim" -> "cutaway"), then the synth."""
        keys = fallback_keys(key)
        if self.assets is not None:
            for k in keys:
                data = self.assets.sound(k)
                if data is not None:
                    return data
        return self.synth(keys[-1]) if self.synth is not None else None

    def _stream_ok(self) -> bool:
        s = self._stream
        if s is None:
            return False
        try:
            return bool(s.active)
        except Exception:  # noqa: BLE001 - a dead stream may raise anything
            return False

    def _watch(self) -> None:
        while not self._stop.wait(WATCHDOG_S):
            if not self._wanted() or self._stream_ok():
                continue
            if self._stream is not None:
                log.warning("audio stream stopped (device unplugged?); reopening")
                self._close_stream()
            self._open()

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                log.exception("error closing the audio stream")

    def _open(self) -> None:
        with self._open_lock:
            if self._stream_ok() or self._stop.is_set():
                return
            try:
                if self.requested == "meeting" and self._open_meeting():
                    return
                self._open_speakers()
            except Exception as e:  # e.g. PortAudio errors while listing devices
                self.mode, self.warning = "none", f"audio unavailable: {e}"
                log.exception("could not open audio")

    def _open_speakers(self) -> None:
        dev = _wasapi_default("output")
        attempts = [(dev, _extra_settings(dev))] + ([(None, None)] if dev is not None else [])
        for device, extra in attempts:
            try:
                stream = sd.OutputStream(
                    samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32", device=device,
                    latency=LATENCY_S, callback=self._out_cb, extra_settings=extra,
                )
                stream.start()
            except Exception as e:  # noqa: BLE001 - try the next device
                log.warning("could not open audio output %s: %s", device, e)
                continue
            self.mode = "speakers"
            self._adopt(stream)
            return
        self.mode = "none"
        self.warning = "no audio output device could be opened; sounds are off"
        log.warning(self.warning)

    def _open_meeting(self) -> bool:
        cable = find_output_device(CABLE_NAME)
        if cable is None:
            self.warning = "VB-CABLE not found: meeting sound needs it, playing on your speakers instead"
            log.warning(self.warning)
            return False
        mic = _wasapi_default("input")
        if mic is None:
            self.warning = "no WASAPI microphone found: playing sounds on your speakers instead"
            log.warning(self.warning)
            return False
        in_ch = max(1, min(2, sd.query_devices(mic)["max_input_channels"]))
        try:
            stream = sd.Stream(
                samplerate=SAMPLE_RATE, channels=(in_ch, CHANNELS), dtype="float32", device=(mic, cable),
                latency=LATENCY_S, callback=self._duplex_cb, extra_settings=_extra_settings(mic, cable),
            )
            stream.start()
        except Exception as e:  # noqa: BLE001 - fall back to speakers
            self.warning = f"could not route sound into the meeting ({e}); playing on your speakers instead"
            log.warning(self.warning)
            return False
        self.mode, self.warning = "meeting", ""
        self._adopt(stream)
        return True

    def _adopt(self, stream) -> None:
        self._stream = stream
        lat = stream.latency
        lat = lat[1] if isinstance(lat, (tuple, list)) else lat
        self._latency = float(lat) if lat else LATENCY_S
        log.info("audio %s: device %s (latency %.0f ms)", self.mode, stream.device, self._latency * 1000)

    def _take_voices(self) -> list[Voice]:
        with self._lock:
            if self._clear:
                self._voices = []
                self._clear = False
            if self._pending:
                self._voices.extend(self._pending)
                self._pending.clear()
            return self._voices

    def _mix(self, outdata: np.ndarray) -> None:
        try:
            left = mix_voices(outdata, self._take_voices(), self.volume)
            with self._lock:
                self._voices = [] if self._clear else left
        except Exception:
            outdata.fill(0)
            if not self._cb_error_logged:
                self._cb_error_logged = True
                log.exception("audio callback failed")

    def _out_cb(self, outdata, frames, time_info, status) -> None:
        outdata.fill(0)
        self._mix(outdata)

    def _duplex_cb(self, indata, outdata, frames, time_info, status) -> None:
        outdata[:] = indata.mean(axis=1, keepdims=True) if indata.shape[1] > 1 else indata
        self._mix(outdata)
