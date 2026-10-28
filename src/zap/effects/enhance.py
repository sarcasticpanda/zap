"""Sharper memes: AI upscaling (Real-ESRGAN) of small meme GIFs, once, in the background.

Meme GIFs are often 250-500 px tall and dithered; blown up to 720p they look
soft and grainy. When ``real_esrgan_x4_fp16.onnx`` is in the models folder
(``zap fetch-models --enhance``, 36 MB) and a GPU is available through
DirectML, each frame of a small meme is upscaled 4x (cleaning up GIF dither and
JPEG blocks), mixed with a little of the original texture, and fitted to the
output size again. That takes ~1-3 s a frame, so it runs on a background
thread; the meme shows the plain frames until the sharp ones are in, and the
result is cached on disk so it happens only once per meme and output size.
"""

from __future__ import annotations

import hashlib
import logging
import queue
import threading
from collections.abc import Callable

import cv2
import numpy as np

from zap import paths
from zap.fetch_models import model_path

log = logging.getLogger(__name__)

MODEL = "real_esrgan_x4_fp16.onnx"
MODEL_X2 = "real_esrgan_x2_fp16.onnx"  # used instead when 2x is enough (faster), if present
TILE, PAD = 256, 16  # upscaled in tiles: little GPU memory, and one fixed shape to compile
SMALL = 0.8  # upscale memes whose source is shorter than this x the output height
TEXTURE = 0.2  # how much of the original (plain-resized) frame is mixed back in, for texture
CACHE_VERSION = 1


def available() -> bool:
    if not model_path(MODEL).is_file():
        return False
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return "DmlExecutionProvider" in ort.get_available_providers()  # on CPU alone it would take minutes a frame


def _cache_path(key: str):
    return paths.home() / "cache" / f"enhanced_{key}.npz"


def cache_key(frames: list[np.ndarray], tag: str) -> str:
    h = hashlib.sha1(f"{tag}:{CACHE_VERSION}:{TEXTURE}".encode())
    for f in (frames[0], frames[len(frames) // 2], frames[-1]):
        h.update(cv2.resize(f, (32, 18), interpolation=cv2.INTER_AREA).tobytes())
    h.update(f"{len(frames)}x{frames[0].shape}".encode())
    return h.hexdigest()[:16]


def load_cached(key: str, n: int) -> list[np.ndarray] | None:
    """The enhanced frames saved by an earlier run, or None."""
    p = _cache_path(key)
    if not p.is_file():
        return None
    try:
        with np.load(p) as data:
            if int(data["version"]) != CACHE_VERSION or len(data["sizes"]) != n:
                return None
            blob, sizes = data["blob"], data["sizes"]
        out, o = [], 0
        for size in sizes:
            out.append(cv2.imdecode(blob[o:o + size], cv2.IMREAD_COLOR))
            o += size
        return out
    except Exception:  # noqa: BLE001 - any damage (e.g. cut off mid-write) just means: compute it again
        log.warning("ignoring a damaged meme cache %s", p.name)
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def _save(key: str, frames: list[np.ndarray]) -> None:
    enc = [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])[1] for f in frames]
    p = _cache_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")  # written aside, then renamed: never a half-written cache
    np.savez(tmp, version=CACHE_VERSION, sizes=np.array([len(e) for e in enc]), blob=np.concatenate(enc))
    tmp.replace(p)


class Enhancer:
    """One background worker that upscales queued memes, frame by frame."""

    def __init__(self) -> None:
        self._jobs: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._sess = None
        self.status = ""

    def submit(self, key: str, sources: list[np.ndarray], present: Callable[[np.ndarray], np.ndarray],
               target: list[np.ndarray], name: str, need: float = 4.0) -> None:
        """Upscale ``sources``, fit each with ``present`` and swap it into ``target[i]`` as it's done."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="zap-enhance", daemon=True)
            self._thread.start()
        self._jobs.put((key, sources, present, target, name, need))

    def _run(self) -> None:
        while True:
            key, sources, present, target, name, need = self._jobs.get()
            try:
                model = MODEL_X2 if need <= 2.2 and model_path(MODEL_X2).is_file() else MODEL
                out = []
                for i, src in enumerate(sources):
                    self.status = f"sharpening the {name} meme {i}/{len(sources)}"
                    f = present(self._upscale(src, model))
                    plain = target[i]
                    if plain.shape == f.shape:
                        f = cv2.addWeighted(f, 1.0 - TEXTURE, plain, TEXTURE, 0)
                    out.append(f)
                for i, f in enumerate(out):  # all at once, so a clip never mixes sharp and soft frames
                    target[i] = f
                _save(key, out)
                log.info("the %s meme is sharpened (%d frames)", name, len(out))
            except Exception:
                log.exception("sharpening the %s meme failed; keeping the plain frames", name)
            finally:
                self.status = ""
                if self._jobs.empty():
                    self._sess = {}  # free the GPU memory until the next meme

    def _upscale(self, bgr: np.ndarray, model: str) -> np.ndarray:
        sess = self._session(model)
        name = sess.get_inputs()[0].name
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        padded = cv2.copyMakeBorder(rgb, PAD, PAD + TILE, PAD, PAD + TILE, cv2.BORDER_REFLECT_101)
        k = None
        out = None
        for y in range(0, h, TILE):
            for x in range(0, w, TILE):
                tile = padded[y:y + TILE + 2 * PAD, x:x + TILE + 2 * PAD].transpose(2, 0, 1)[None]
                up = sess.run(None, {name: np.ascontiguousarray(tile)})[0][0].transpose(1, 2, 0)
                if out is None:
                    k = up.shape[0] // (TILE + 2 * PAD)
                    out = np.zeros((h * k, w * k, 3), np.float32)
                th, tw = min(TILE, h - y), min(TILE, w - x)
                out[y * k:(y + th) * k, x * k:(x + tw) * k] = up[PAD * k:(PAD + th) * k, PAD * k:(PAD + tw) * k]
        return cv2.cvtColor(np.clip(out * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _session(self, model: str):
        if not isinstance(self._sess, dict):
            self._sess = {}
        if model not in self._sess:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.log_severity_level = 3
            self._sess[model] = ort.InferenceSession(str(model_path(model)), so,
                                                     providers=["DmlExecutionProvider", "CPUExecutionProvider"])
        return self._sess[model]


_ENHANCER: Enhancer | None = None


def enhancer() -> Enhancer:
    global _ENHANCER
    if _ENHANCER is None:
        _ENHANCER = Enhancer()
    return _ENHANCER
