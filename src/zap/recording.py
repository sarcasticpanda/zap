"""Record a session's gesture data, and replay it deterministically.

``zap record`` exists so a gesture that misfires (or never fires) for someone
can be studied and tuned offline: it saves what the detectors saw - hand and
face landmarks, blendshapes, the microphone - plus a small 640x360 video for
reference, into ``recordings/<timestamp>/``. It runs only when the user starts
it on purpose; nothing else in ZAP writes camera frames or audio to disk.

``zap replay <dir>`` feeds the saved signals and audio through the same
PoseDetector / Arbiter / snap detectors with a ReplayClock, so the same
recording always prints the same events.

Per-frame records (``signals.jsonl``, one JSON object per line) use the same
format as the landmark fixtures in ``tests/fixtures_local``::

    {"t": s since start, "seq": n, "step": "", "hands": [{"lm": 21x3 px, "score", "g", "gs"}],
     "face": null | {"lm": 478x2 px, "fresh", "blend": {name: score}, "matrix": 4x4 | null}}
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import wave
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from zap.signals import Face, Hand, Signals

log = logging.getLogger(__name__)

AUDIO_RATE = 48000
VIDEO_SIZE = (640, 360)
AUDIO_BLOCK = 480  # 10 ms blocks when replaying audio


# ---------------------------------------------------------------- per-frame records
def hand_record(h: Hand) -> dict[str, Any]:
    return {"lm": np.round(h.lm, 1).tolist(), "score": round(float(h.score), 3), "g": h.gesture,
            "gs": round(float(h.gesture_score), 3)}


def face_record(f: Face) -> dict[str, Any]:
    return {
        "lm": np.round(f.lm[:, :2], 1).tolist(),
        "fresh": bool(f.fresh),
        "blend": {k: round(float(v), 3) for k, v in f.blend.items()},
        "matrix": None if f.matrix is None else np.round(f.matrix, 4).tolist(),
    }


def frame_record(sig: Signals, t: float, step: str = "") -> dict[str, Any]:
    """One frame's gesture inputs; ``t`` is seconds since the recording started."""
    return {
        "t": round(float(t), 4),
        "seq": int(sig.seq),
        "step": step,
        "hands": [hand_record(h) for h in sig.hands],
        "face": None if sig.face is None else face_record(sig.face),
    }


def signals_from_record(rec: dict[str, Any], seq: int = 0, width: int = 1280, height: int = 720) -> Signals:
    """Rebuild Signals from a frame record (face z is not stored and comes back as 0)."""
    t = float(rec["t"])
    hands = [
        Hand(np.asarray(h["lm"], np.float32).reshape(21, 3), float(h["score"]), h.get("g", "None"),
             float(h.get("gs", 0.0)))
        for h in rec.get("hands") or []
    ]
    face = None
    f = rec.get("face")
    if f is not None:
        xy = np.asarray(f["lm"], np.float32)
        lm = np.zeros((len(xy), 3), np.float32)
        lm[:, :2] = xy[:, :2]
        m = f.get("matrix")
        face = Face(lm, dict(f.get("blend") or {}), None if m is None else np.asarray(m, np.float64), t,
                    bool(f.get("fresh", True)))
    return Signals(t, int(rec.get("seq", seq)), hands, face, None, False, width, height)


def load_frames(path: Path) -> list[dict[str, Any]]:
    """Frame records from a JSON list (fixtures) or JSON lines (``signals.jsonl``)."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------- audio
class WavTap:
    """Stands in for a SnapAudioDetector inside MicStream and writes the microphone to a WAV.

    MicStream hands over runs of contiguous samples with their capture time; gaps
    (dropped blocks) are filled with silence so the file stays aligned to the clock.
    """

    def __init__(self, path: Path, rate: int = AUDIO_RATE) -> None:
        self.rate = rate
        self.t0: float | None = None  # perf_counter time of the first sample
        self.samples = 0
        self.pending_onset = None
        self.noise_floor_dbfs = -100.0
        self._lock = threading.Lock()
        self._wav = wave.open(str(path), "wb")  # noqa: SIM115 - closed in close(), written from another thread
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(rate)

    def gate(self, until: float) -> None:  # MicStream API; nothing to gate here
        pass

    def process(self, block: np.ndarray, t_start: float) -> list:
        with self._lock:
            if self._wav is None:
                return []
            if self.t0 is None:
                self.t0 = t_start
            gap = round((t_start - self.t0) * self.rate) - self.samples
            if 0 < gap < self.rate:  # dropped audio: pad so later samples keep their time
                self._write(np.zeros(gap, np.float32))
            self._write(np.asarray(block, np.float32))
        return []

    def _write(self, x: np.ndarray) -> None:
        self._wav.writeframes((np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes())
        self.samples += x.size

    def close(self) -> None:
        with self._lock:
            if self._wav is not None:
                self._wav.close()
                self._wav = None


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        rate, n, width, ch = w.getframerate(), w.getnframes(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(n)
    if width != 2:
        raise ValueError(f"{path.name}: expected 16-bit audio")
    x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    return x.reshape(-1, ch).mean(axis=1), rate


# ---------------------------------------------------------------- record
def record(cfg, seconds: float = 30.0, preview: bool = True, out_dir: Path | None = None) -> Path:
    """Record camera landmarks, microphone and a small reference video. Returns the folder."""
    import cv2

    from zap import paths
    from zap.devices.audio_in import MicStream
    from zap.devices.camera import Camera
    from zap.vision.perception import Perception

    folder = Path(out_dir) if out_dir else paths.recordings_dir() / time.strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    W, H = cfg.output.width, cfg.output.height
    perception = Perception(cfg.perception, W, H)
    cam = Camera.from_config(cfg).start()
    tap = WavTap(folder / "audio.wav")
    mic = MicStream(tap, device=cfg.snap.mic or None)  # type: ignore[arg-type]
    mic_ok = mic.start()
    video = cv2.VideoWriter(str(folder / "video.avi"), cv2.VideoWriter_fourcc(*"MJPG"), float(cfg.camera.fps),
                            VIDEO_SIZE)
    small = np.empty((VIDEO_SIZE[1], VIDEO_SIZE[0], 3), np.uint8)
    t0 = time.perf_counter()
    n = 0
    try:
        with open(folder / "signals.jsonl", "w", encoding="utf-8") as sig_f, \
                open(folder / "frames.jsonl", "w", encoding="utf-8") as frames_f:
            while time.perf_counter() - t0 < seconds:
                pkt = cam.wait_latest(1.0)
                if pkt is None:
                    continue
                sig = perception.run(pkt, need_face=True, need_mask=False, need_hands=True)
                cv2.resize(pkt.frame, VIDEO_SIZE, dst=small, interpolation=cv2.INTER_AREA)
                video.write(small)
                t = pkt.t - t0
                frames_f.write(json.dumps({"seq": n, "t": round(t, 4)}) + "\n")
                sig_f.write(json.dumps(frame_record(sig, t)) + "\n")
                n += 1
                if preview:
                    view = cv2.flip(small, 1) if cfg.app.preview_mirror else small.copy()
                    left = seconds - (time.perf_counter() - t0)
                    cv2.circle(view, (20, 20), 8, (0, 0, 255), -1)
                    cv2.putText(view, f"REC {left:4.1f} s  (q = stop)", (36, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow("ZAP record", view)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
    finally:
        mic.stop()
        tap.close()
        video.release()
        cam.stop()
        perception.close()
        if preview:
            cv2.destroyAllWindows()
    meta = {
        "frames": n, "seconds": round(time.perf_counter() - t0, 2), "width": W, "height": H,
        "audio": bool(mic_ok and tap.t0 is not None),
        "audio_t0": None if tap.t0 is None else round(tap.t0 - t0, 5),
        "audio_rate": AUDIO_RATE, "camera": cam.info.name if cam.info else "", "mic": mic.device_name,
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("recorded %d frames into %s", n, folder)
    return folder


# ---------------------------------------------------------------- replay
@dataclass
class ReplayEvent:
    t: float
    kind: str  # an Intent kind, or "snap" for a confirmed snap onset
    gesture: str = ""
    step: str = ""


@dataclass
class ReplayReport:
    frames: int = 0
    seconds: float = 0.0
    audio: bool = False
    events: list[ReplayEvent] = field(default_factory=list)
    snap_candidates: Counter = field(default_factory=Counter)  # audio verdict -> count
    fusion_rejects: Counter = field(default_factory=Counter)

    def counts(self) -> Counter:
        return Counter(f"{e.kind}:{e.gesture}" if e.gesture else e.kind for e in self.events)

    def format(self) -> str:
        lines = [f"{self.frames} frames, {self.seconds:.1f} s, audio {'yes' if self.audio else 'no'}"]
        for e in self.events:
            lines.append(f"  {e.t:8.3f}  {e.kind:<9} {e.gesture:<12} {e.step}")
        lines.append("counts: " + (", ".join(f"{k} {v}" for k, v in sorted(self.counts().items())) or "none"))
        if self.snap_candidates:
            lines.append("audio candidates: " + ", ".join(f"{k} {v}" for k, v in self.snap_candidates.most_common()))
        if self.fusion_rejects:
            lines.append("snaps rejected by vision: " + ", ".join(
                f"{k} {v}" for k, v in self.fusion_rejects.most_common()))
        return "\n".join(lines)


def _audio_blocks(x: np.ndarray, rate: int, t0: float) -> Iterator[tuple[float, np.ndarray]]:
    for i in range(0, x.size, AUDIO_BLOCK):
        yield t0 + i / rate, x[i:i + AUDIO_BLOCK]


def replay_frames(frames: Iterable[dict[str, Any]], cfg=None, audio: np.ndarray | None = None,
                  audio_rate: int = AUDIO_RATE, audio_t0: float = 0.0, width: int = 1280,
                  height: int = 720) -> ReplayReport:
    """Run frame records (and optional mono audio aligned to the same clock) through the detectors.

    Gestures are armed with no grace period and no leave guard: this is analysis, not a meeting.
    """
    from zap import config as zap_config
    from zap.clock import ReplayClock
    from zap.gestures.arbiter import Arbiter
    from zap.gestures.poses import PoseDetector
    from zap.gestures.snap import BurstCounter, SnapFusion
    from zap.gestures.snap_audio import SnapAudioDetector

    cfg = cfg or zap_config.defaults()
    cfg.app.armed_on_start = True
    cfg.app.arm_grace_s = 0.0
    cfg.zoom.leave_guard_s = 0.0
    frames = list(frames)
    report = ReplayReport(frames=len(frames), audio=audio is not None)
    if not frames:
        return report
    clock = ReplayClock(float(frames[0]["t"]))
    det = PoseDetector(cfg.gestures)
    arb = Arbiter(cfg, clock)
    vision_only = audio is None or cfg.snap.vision_only
    gap = cfg.snap.burst_gap_ms / 1000.0
    fusion = SnapFusion(av_offset_s=cfg.snap.av_offset_ms / 1000.0, vision_only=vision_only, burst_gap_s=gap)
    bursts = BurstCounter(gap_s=gap, max_count=1, clamp=True) if vision_only else BurstCounter(gap_s=gap)
    audio_det = SnapAudioDetector(rate=audio_rate, sensitivity=cfg.snap.sensitivity)
    blocks = _audio_blocks(audio, audio_rate, audio_t0) if audio is not None else iter(())
    next_block = next(blocks, None)
    for i, rec in enumerate(frames):
        sig = signals_from_record(rec, i, width, height)
        t = sig.t
        clock.set(t)
        while next_block is not None and next_block[0] + next_block[1].size / audio_rate <= t:
            for c in audio_det.process(next_block[1], next_block[0]):
                report.snap_candidates[c.reason] += 1
                fusion.add_candidate(c)
            next_block = next(blocks, None)
        fusion.add_frame(sig)
        events: list[Any] = []
        fusion.rejects.clear()  # so everything in it afterwards was decided this frame
        for t_on, where in fusion.resolve(t, in_burst=bursts.in_burst):
            report.events.append(ReplayEvent(round(t_on, 3), "snap", "", rec.get("step", "")))
            marker = bursts.add(t_on, where)
            if marker is not None:
                events.append(marker)
        report.fusion_rejects.update(fusion.rejects)
        done, markers = bursts.tick(t, audio_det.pending_onset, fusion.pending_onset)
        events += done + markers
        truths = det.update(sig, t)
        arb.presence_state = "visible"
        for it in arb.update(truths, events, t, busy=False, snap_loaded=det.snap_loaded, t_on=det.t_on):
            report.events.append(ReplayEvent(round(it.t, 3), it.kind, it.gesture, rec.get("step", "")))
    report.seconds = float(frames[-1]["t"]) - float(frames[0]["t"])
    return report


def replay(folder: Path, cfg=None) -> ReplayReport:
    """Replay a ``zap record`` folder (signals.jsonl + audio.wav + meta.json)."""
    folder = Path(folder)
    sig_path = folder / "signals.jsonl"
    if not sig_path.is_file():
        raise FileNotFoundError(f"{sig_path} not found (is this a 'zap record' folder?)")
    meta = {}
    if (folder / "meta.json").is_file():
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    audio, rate, t0 = None, AUDIO_RATE, 0.0
    if (folder / "audio.wav").is_file() and meta.get("audio_t0") is not None:
        audio, rate = read_audio(folder / "audio.wav")
        t0 = float(meta["audio_t0"])
        if not math.isfinite(t0) or audio.size == 0:
            audio = None
    return replay_frames(load_frames(sig_path), cfg, audio, rate, t0, int(meta.get("width", 1280)),
                         int(meta.get("height", 720)))
