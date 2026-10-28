"""Your face on a meme actor (the Harvey cutaway), drawn by InsightFace's inswapper.

inswapper is the identity-swap network behind FaceFusion, roop and
Deep-Live-Cam. Given a 128x128 aligned face and a 512-d ArcFace identity, it
redraws the face with that identity while keeping the pose, expression and
lighting of the original. ZAP learns your identity from a few good frames of
the live camera (facing the camera, big enough, sharp), then swaps every frame
of the meme clip once, in the background. Playing the clip costs nothing.

Optional models, fetched by ``zap fetch-models --faceswap`` (~450 MB;
InsightFace's models are licensed for non-commercial research use only):

* ``arcface_w600k_r50.onnx``: who you are (identity from a 112x112 aligned face)
* ``inswapper_128_fp16.onnx``: redraws a 128x128 aligned face with an identity

Runs on the GPU through DirectML when there is one (~30 ms a frame), else on
the CPU (~2 s a frame: the clip is ready about a minute after ZAP first sees you).
Without the models, face-swap cutaways play the original clip.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from zap.fetch_models import model_path

log = logging.getLogger(__name__)

ARCFACE = "arcface_w600k_r50.onnx"
SWAPPER = "inswapper_128_fp16.onnx"
EMAP = "inswapper_emap.npy"  # the swapper's identity projection, extracted once from the model

# ArcFace's canonical 5 points (eye centres, nose tip, mouth corners) in a 112x112 crop.
ARC_DST = np.float32([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                      [41.5493, 92.3655], [70.7299, 92.2041]])

# Learning the user's identity.
SAMPLES = 5  # embeddings averaged into one identity
SAMPLE_EVERY_S = 0.5
MIN_FACE_PX = 90  # face width in the frame
FRONTAL_MAX = 0.14  # nose offset from the eye midpoint, in eye distances
MIN_SHARPNESS = 3.5  # variance of the Laplacian on the face crop (webcams are soft; this only drops motion blur)


def available() -> bool:
    return model_path(ARCFACE).is_file() and model_path(SWAPPER).is_file()


def five_points(lm: np.ndarray) -> np.ndarray:
    """ArcFace's 5 points from MediaPipe's 478: eye centres and mouth corners in image order (left first)."""
    p = np.asarray(lm, np.float32)[:, :2]
    eyes = sorted((p[[33, 133]].mean(axis=0), p[[362, 263]].mean(axis=0)), key=lambda q: q[0])
    mouth = sorted((p[61], p[291]), key=lambda q: q[0])
    return np.float32([eyes[0], eyes[1], p[1], mouth[0], mouth[1]])


def is_frontal(lm: np.ndarray, limit: float = FRONTAL_MAX) -> bool:
    k = five_points(lm)
    axis = k[1] - k[0]
    d = float(np.linalg.norm(axis))
    if d < 1:
        return False
    off = float(np.dot(k[2] - (k[0] + k[1]) / 2, axis / d)) / d
    return abs(off) < limit


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity (2x3) taking ``src`` points onto ``dst``."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - ms, dst - md
    u, sv, vt = np.linalg.svd(dc.T @ sc / len(src))
    d = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    r = u @ d @ vt
    scale = float(np.trace(np.diag(sv) @ d)) / max(float((sc ** 2).sum()) / len(src), 1e-9)
    t = md - scale * r @ ms
    return np.hstack([scale * r, t[:, None]]).astype(np.float32)


class Swapper:
    """The two ONNX models, loaded on first use (GPU through DirectML when available)."""

    def __init__(self) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = 4  # CPU fallback: leave cores for the camera pipeline
        providers = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.arc = ort.InferenceSession(str(model_path(ARCFACE)), so, providers=providers)
        self.swap_sess = ort.InferenceSession(str(model_path(SWAPPER)), so, providers=providers)
        self.device = "GPU" if self.swap_sess.get_providers()[0] == "DmlExecutionProvider" else "CPU"
        self.emap = _emap()

    def embed(self, bgr: np.ndarray, lm: np.ndarray) -> np.ndarray:
        """Unit-length ArcFace identity of the face at ``lm``."""
        M = umeyama(five_points(lm), ARC_DST)
        crop = cv2.warpAffine(bgr, M, (112, 112), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
        blob = cv2.dnn.blobFromImage(crop, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        e = self.arc.run(None, {self.arc.get_inputs()[0].name: blob})[0][0]
        return e / max(float(np.linalg.norm(e)), 1e-9)

    def latent(self, identity: np.ndarray) -> np.ndarray:
        v = identity.reshape(1, -1) @ self.emap
        return (v / max(float(np.linalg.norm(v)), 1e-9)).astype(np.float32)

    def swap(self, bgr: np.ndarray, lm: np.ndarray, latent: np.ndarray) -> np.ndarray:
        """A copy of ``bgr`` with the face at ``lm`` redrawn with identity ``latent``."""
        dst = ARC_DST + np.float32([8.0, 0.0])  # the 128 crop is the 112 template, shifted 8 px right
        M = umeyama(five_points(lm), dst)
        crop = cv2.warpAffine(bgr, M, (128, 128), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        blob = cv2.dnn.blobFromImage(crop, 1 / 255.0, (128, 128), (0, 0, 0), swapRB=True)
        pred = self.swap_sess.run(None, {"target": blob, "source": latent})[0][0]
        face = cv2.cvtColor(np.clip(pred.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        face = cv2.addWeighted(face, 1.4, cv2.GaussianBlur(face, (0, 0), 1.0), -0.4, 0)  # 128 px is soft once enlarged
        return _paste(bgr, face, M, lm)


# MediaPipe's face outline; the swapped face is pasted inside it so the actor keeps his own hair.
FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)


def _paste(bgr: np.ndarray, face: np.ndarray, M: np.ndarray, lm: np.ndarray) -> np.ndarray:
    """Put the 128x128 aligned ``face`` back where crop transform ``M`` took it from, inside the face
    outline of ``lm``, feathered."""
    H, W = bgr.shape[:2]
    oval = np.asarray(lm, np.float32)[list(FACE_OVAL), :2]
    size = float(max(np.ptp(oval[:, 0]), np.ptp(oval[:, 1])))
    pad = int(0.15 * size)
    x0, y0 = (np.maximum(oval.min(axis=0) - pad, 0)).astype(int)
    x1, y1 = (np.minimum(oval.max(axis=0) + pad, (W, H))).astype(int)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return bgr.copy()
    IM = cv2.invertAffineTransform(M)
    IM[:, 2] -= (x0, y0)  # work in the face's bounding box only
    size_r = (x1 - x0, y1 - y0)
    back = cv2.warpAffine(face, IM, size_r, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    crop_mask = cv2.warpAffine(np.full((128, 128), 255, np.uint8), IM, size_r, flags=cv2.INTER_LINEAR)
    outline = np.zeros(back.shape[:2], np.uint8)
    cv2.fillConvexPoly(outline, cv2.convexHull((oval - (x0, y0)).astype(np.int32)), 255, cv2.LINE_AA)
    k = max(3, int(0.05 * size))
    mask = cv2.min(cv2.erode(crop_mask, np.ones((k, k), np.uint8)), cv2.erode(outline, np.ones((k, k), np.uint8)))
    b = max(2, int(0.04 * size))
    a = cv2.GaussianBlur(mask, (2 * b + 1, 2 * b + 1), 0).astype(np.float32) * (1 / 255)
    out = bgr.copy()
    roi = out[y0:y1, x0:x1]
    cv2.blendLinear(back, roi.copy(), a, 1.0 - a, dst=roi)
    return out


def _emap() -> np.ndarray:
    p = model_path(EMAP)
    if p.is_file():
        return np.load(p)
    import onnx
    from onnx import numpy_helper

    e = numpy_helper.to_array(onnx.load(str(model_path(SWAPPER))).graph.initializer[-1]).astype(np.float32)
    np.save(p, e)
    return e


# ---------------------------------------------------------------- the service
@dataclass
class _Clip:
    frames: list[np.ndarray]
    pts: Callable[[int], np.ndarray | None]  # landmarks of frame i, or None (no face found)
    ready: threading.Event
    swapped: list[np.ndarray] | None = None
    done: int = 0


@dataclass
class _State:
    samples: list[np.ndarray] = field(default_factory=list)
    identity: np.ndarray | None = None
    last_offer: float = -1e9
    busy: bool = False
    error: str = ""


class FaceSwapService:
    """Learns your identity from the live camera and swaps registered meme clips, on one worker thread."""

    def __init__(self) -> None:
        self._s = _State()
        self._clips: dict[str, _Clip] = {}
        self._jobs: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._swapper: Swapper | None = None
        self._thread: threading.Thread | None = None
        self.enabled = available()

    # ---------------------------------------------------------- inputs
    def register(self, key: str, frames: list[np.ndarray], pts: Callable[[int], np.ndarray | None],
                 ready: threading.Event) -> None:
        """A clip to show with the user's face; swapped once the identity and the clip's landmarks are known."""
        if not self.enabled:
            return
        with self._lock:
            if key in self._clips:
                return
            self._clips[key] = _Clip(frames, pts, ready)
        if self._s.identity is not None:
            self._submit(("clip", key))

    def offer(self, frame: np.ndarray, lm: np.ndarray | None, now: float) -> None:
        """A live frame; cheap unless it is used as an identity sample (then copied to the worker)."""
        s = self._s
        if not self.enabled or s.identity is not None or s.busy or lm is None or now - s.last_offer < SAMPLE_EVERY_S:
            return
        k = five_points(lm)
        width = float(np.ptp(np.asarray(lm)[:, 0]))
        if width < MIN_FACE_PX or not is_frontal(lm):
            return
        x0, y0 = np.maximum(k.min(axis=0) - 0.2 * width, 0).astype(int)
        x1, y1 = (k.max(axis=0) + 0.2 * width).astype(int)
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY) if x1 > x0 + 8 and y1 > y0 + 8 else None
        if gray is None or cv2.Laplacian(gray, cv2.CV_32F).var() < MIN_SHARPNESS:
            return
        s.last_offer, s.busy = now, True
        self._submit(("sample", frame.copy(), np.asarray(lm, np.float32)[:, :2].copy()))

    def forget(self) -> None:
        """Drop the learnt identity and swapped clips (learn again from the next good frames)."""
        with self._lock:
            self._s = _State()
            for c in self._clips.values():
                c.swapped, c.done = None, 0

    # ---------------------------------------------------------- outputs
    def swapped(self, key: str) -> list[np.ndarray] | None:
        c = self._clips.get(key)
        return c.swapped if c is not None else None

    @property
    def status(self) -> str:
        s = self._s
        if not self.enabled:
            return ""
        if s.error:
            return f"face swap off: {s.error}"
        if s.identity is None:
            return f"learning your face {len(s.samples)}/{SAMPLES} (look at the camera)"
        pending = [c for c in self._clips.values() if c.swapped is None]
        if pending:
            c = pending[0]
            return f"putting your face on the meme {c.done}/{len(c.frames)}"
        return ""

    # ---------------------------------------------------------- worker
    def _submit(self, job: tuple) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="zap-faceswap", daemon=True)
            self._thread.start()
        self._jobs.put(job)

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if self._swapper is None:
                    self._swapper = Swapper()
                    log.info("face swap models loaded (%s)", self._swapper.device)
                if job[0] == "sample":
                    self._sample(job[1], job[2])
                elif job[0] == "clip":
                    self._swap_clip(job[1])
            except Exception as e:  # never take the app down for a cosmetic feature
                log.exception("face swap failed")
                self._s.error = str(e)[:80]
                self.enabled = False
                return
            finally:
                self._s.busy = False

    def _sample(self, frame: np.ndarray, lm: np.ndarray) -> None:
        s = self._s
        e = self._swapper.embed(frame, lm)
        if s.samples:
            mean = np.mean(s.samples, axis=0)
            if float(np.dot(e, mean / np.linalg.norm(mean))) < 0.35:  # someone else, or a bad frame
                return
        s.samples.append(e)
        if len(s.samples) >= SAMPLES:
            ident = np.mean(s.samples, axis=0)
            s.identity = ident / np.linalg.norm(ident)
            log.info("learnt your face for the face-swap memes")
            for key in list(self._clips):
                self._jobs.put(("clip", key))

    def _swap_clip(self, key: str) -> None:
        c = self._clips.get(key)
        ident = self._s.identity
        if c is None or c.swapped is not None or ident is None:
            return
        c.ready.wait()
        latent = self._swapper.latent(ident)
        out = []
        for i, f in enumerate(c.frames):
            p = c.pts(i)
            out.append(self._swapper.swap(f, p, latent) if p is not None else f)
            c.done = i + 1
        if self._s.identity is ident:  # not forgotten meanwhile
            c.swapped = out
            log.info("your face is ready on %s (%d frames)", key, len(out))


_SERVICE: FaceSwapService | None = None


def service() -> FaceSwapService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = FaceSwapService()
    return _SERVICE
