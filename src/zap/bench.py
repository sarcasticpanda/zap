"""Measure what perception costs on this machine, with the real webcam.

    python -m zap.bench [--seconds 20] [--sizes 640x360,960x540] [--camera INDEX]

For each inference size it prints p50/p95 per model when the three MediaPipe
tasks run one after another (sequential) versus side by side on the pool
(pooled), the whole-perception p95 at the configured face/mask cadence, how
many frames had two hands or a face, and the fps the camera really delivered.
This is how to choose ``[perception]`` settings for a slower laptop. Frames
only ever live in memory.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass, field

import cv2
import numpy as np

from zap.config import PerceptionCfg, defaults
from zap.signals import FramePacket
from zap.vision.perception import TASK_KEYS, Perception

log = logging.getLogger(__name__)

OUT_W, OUT_H, CAM_FPS = 1280, 720, 30
_VIRTUAL_HINTS = ("virtual", "obs")


# ---------------------------------------------------------------- camera
def pick_physical(cameras: list[tuple[int, str]]) -> int | None:
    """First DirectShow camera whose name doesn't look like a virtual camera."""
    for index, name in cameras:
        if not any(h in (name or "").lower() for h in _VIRTUAL_HINTS):
            return index
    return None


BACKENDS = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF}


def list_cameras(backend: str = "DSHOW") -> list[tuple[int, str]]:
    try:
        from cv2_enumerate_cameras import enumerate_cameras

        return [(c.index, c.name or "") for c in enumerate_cameras(BACKENDS[backend])]
    except Exception:  # enumeration is best effort
        log.warning("could not enumerate %s cameras", backend, exc_info=True)
        return []


def camera_candidates(listings: dict[str, list[tuple[int, str]]]) -> list[tuple[str, int]]:
    """(backend, index) to try in order: the first non-virtual camera of each backend.

    DirectShow comes first, but some webcams appear there with an empty name and
    refuse to open while Media Foundation opens them fine, so MSMF is the fallback.
    """
    out = []
    for backend in BACKENDS:
        index = pick_physical(listings.get(backend, []))
        if index is not None:
            out.append((backend, index))
    return out


def try_open(backend: str, index: int) -> cv2.VideoCapture | None:
    """Open as MJPG 1280x720@30 and read one frame; None if the camera won't deliver."""
    cap = cv2.VideoCapture(index, BACKENDS[backend])
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, OUT_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, OUT_H)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        ok, _ = cap.read()
        if ok:
            return cap
    cap.release()
    return None


def open_camera(
    candidates: list[tuple[str, int]], budget_s: float = 120.0, wait_s: float = 5.0
) -> tuple[cv2.VideoCapture, str, int]:
    """Try each candidate in turn, retrying for up to ``budget_s`` while the camera is busy."""
    if not candidates:
        raise RuntimeError("no physical camera found (only virtual ones)")
    deadline = time.perf_counter() + budget_s
    while True:
        for backend, index in candidates:
            cap = try_open(backend, index)
            if cap is not None:
                return cap, backend, index
            print(f"{backend} camera {index} did not deliver frames")
        if time.perf_counter() + wait_s > deadline:
            raise RuntimeError(f"could not read frames from any of {candidates}")
        print(f"camera busy? retrying in {wait_s:.0f}s ...")
        time.sleep(wait_s)


def _fourcc(cap: cv2.VideoCapture) -> str:
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    text = "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))
    return text.strip() if text.isprintable() and text.strip() else "?"  # MSMF reports no FourCC


class Grabber:
    """Reads the camera on its own thread so camera fps is measured independently of processing."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self.cap = cap
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._t = 0.0
        self.seq = 0
        self._stop = False
        self.count = 0
        self._t_start = 0.0
        self._t_end: float | None = None
        self._thread = threading.Thread(target=self._loop, name="zap-bench-grab", daemon=True)

    def start(self) -> Grabber:
        self._t_start = time.perf_counter()
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop:
            ok, frame = self.cap.read()
            t = time.perf_counter()
            if not ok:
                time.sleep(0.005)
                continue
            if frame.shape[1] != OUT_W or frame.shape[0] != OUT_H:
                frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
            with self._cond:
                self._frame, self._t = frame, t
                self.seq += 1
                self.count += 1
                self._cond.notify_all()

    def next(self, after: int, timeout: float = 1.0) -> FramePacket | None:
        with self._cond:
            if not self._cond.wait_for(lambda: self.seq > after, timeout):
                return None
            return FramePacket(frame=self._frame, t=self._t, seq=self.seq)

    def fps(self) -> float:
        dt = (self._t_end or time.perf_counter()) - self._t_start
        return self.count / dt if dt > 0 else 0.0

    def stop(self) -> None:
        self._stop = True
        self._t_end = time.perf_counter()
        self._thread.join(timeout=2.0)


# ---------------------------------------------------------------- measurement
@dataclass
class Phase:
    ms: dict[str, list[float]] = field(default_factory=lambda: {k: [] for k in (*TASK_KEYS, "total")})
    frames: int = 0

    def pct(self, key: str, q: float) -> float | None:
        v = self.ms[key]
        return float(np.percentile(v, q)) if v else None


@dataclass
class SizeResult:
    size: tuple[int, int]
    phases: dict[str, Phase] = field(default_factory=dict)
    frames: int = 0
    any_hand: int = 0
    two_hands: int = 0
    face_runs: int = 0
    faces: int = 0
    polarity: str = "undecided"
    camera_fps: float = 0.0
    sample: str = ""


def _run_phase(p: Perception, grab: Grabber, seconds: float, need_all: bool, res: SizeResult) -> Phase:
    phase = Phase()
    seq = grab.seq
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pkt = grab.next(seq)
        if pkt is None:
            continue
        seq = pkt.seq
        sig = p.run(pkt, need_face=need_all, need_mask=need_all)
        for key in p.ran:
            phase.ms[key].append(p.timings[key])
        phase.ms["total"].append(p.timings["total"])
        phase.frames += 1
        res.frames += 1
        res.any_hand += len(sig.hands) >= 1
        res.two_hands += len(sig.hands) >= 2
        if "face" in p.ran:
            res.face_runs += 1
            res.faces += sig.face is not None and sig.face.fresh
        if sig.face is not None and sig.mask is not None and sig.face.fresh and sig.mask_fresh:
            f = sig.face
            res.sample = (
                f"face lm {f.lm.shape}, {len(f.blend)} blendshapes, matrix "
                f"{None if f.matrix is None else f.matrix.shape}; mask {sig.mask.shape} "
                f"mean {float(sig.mask.mean()):.2f}"
            )
    return phase


def _fmt(phase: Phase, key: str) -> str:
    a, b = phase.pct(key, 50), phase.pct(key, 95)
    return "      -      " if a is None else f"{a:5.1f} / {b:5.1f}"


def print_table(results: list[SizeResult], face_every: int, mask_every: int) -> None:
    head = f"{'size':<9} {'mode':<16} {'hands p50/p95':>13}  {'face p50/p95':>13}  {'mask p50/p95':>13}  {'total p50/p95':>13}"
    print(head)
    print("-" * len(head))
    for r in results:
        size = f"{r.size[0]}x{r.size[1]}"
        for name, phase in r.phases.items():
            label = name if name != "cadence" else f"cadence f/{face_every} m/{mask_every}"
            cols = "  ".join(_fmt(phase, k) for k in (*TASK_KEYS, "total"))
            print(f"{size:<9} {label:<16} {cols}   ({phase.frames} frames)")
        print(
            f"{'':<9} hands: >=1 in {r.any_hand}, 2 in {r.two_hands} of {r.frames} frames; face in {r.faces}/{r.face_runs} face runs; "
            f"mask polarity {r.polarity}; camera {r.camera_fps:.1f} fps"
        )
        if r.sample:
            print(f"{'':<9} sample: {r.sample}")
    print("(milliseconds; 'total' is the whole Perception.run including resize; pooled = 3 worker threads)")


def run(
    seconds: float = 20,
    sizes: tuple[tuple[int, int], ...] = ((640, 360), (960, 540)),
    camera: int | None = None,
    backend: str = "DSHOW",
) -> list[SizeResult]:
    """Benchmark perception on the live camera for about ``seconds`` and print a table.

    ``camera`` forces one index on ``backend``; by default the first non-virtual
    camera is used, DirectShow first and Media Foundation as the fallback.
    """
    if camera is None:
        listings = {b: list_cameras(b) for b in BACKENDS}
        candidates = camera_candidates(listings)
        print(f"cameras: {listings} -> trying {candidates}")
    else:
        candidates = [(backend.upper(), camera)]

    base = defaults().perception
    results: list[SizeResult] = []
    with ExitStack() as stack:
        perceptions = []
        for w, h in sizes:  # build (and warm up) before opening the camera
            cfg = PerceptionCfg(infer_width=w, infer_height=h, face_every=base.face_every, mask_every=base.mask_every)
            perceptions.append(stack.enter_context(Perception(cfg, OUT_W, OUT_H)))
        cap, used_backend, used_index = open_camera(candidates)
        stack.callback(cap.release)
        print(
            f"{used_backend} camera {used_index}: "
            f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
            f"{_fourcc(cap)}, {cap.get(cv2.CAP_PROP_FPS):.0f} fps requested"
        )
        grab = Grabber(cap).start()
        stack.callback(grab.stop)
        per_phase = seconds / (3 * len(sizes))
        for p, size in zip(perceptions, sizes):
            res = SizeResult(size=size)
            fps_t0, fps_n0 = time.perf_counter(), grab.count
            p.pooled = False
            res.phases["sequential"] = _run_phase(p, grab, per_phase, True, res)
            p.pooled = True
            res.phases["pooled"] = _run_phase(p, grab, per_phase, True, res)
            res.phases["cadence"] = _run_phase(p, grab, per_phase, False, res)
            res.camera_fps = (grab.count - fps_n0) / (time.perf_counter() - fps_t0)
            res.polarity = p.mask_polarity
            results.append(res)
    print_table(results, base.face_every, base.mask_every)
    print(f"camera overall: {grab.fps():.1f} fps")
    return results


def _parse_sizes(text: str) -> tuple[tuple[int, int], ...]:
    out = []
    for part in text.split(","):
        w, h = part.lower().split("x")
        out.append((int(w), int(h)))
    return tuple(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m zap.bench", description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--sizes", type=_parse_sizes, default=((640, 360), (960, 540)), help="e.g. 640x360,960x540")
    ap.add_argument("--camera", type=int, default=None, help="camera index (default: first non-virtual)")
    ap.add_argument("--backend", choices=[b.lower() for b in BACKENDS], default="dshow", help="for --camera")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(args.seconds, args.sizes, args.camera, args.backend)


if __name__ == "__main__":
    main()
