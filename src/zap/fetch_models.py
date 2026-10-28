"""Download the MediaPipe models ZAP needs into the ZAP home folder."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

from zap import paths

BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "gesture_recognizer.task": "gesture_recognizer/gesture_recognizer/float16/{v}/gesture_recognizer.task",
    "face_landmarker.task": "face_landmarker/face_landmarker/float16/{v}/face_landmarker.task",
    "selfie_segmenter_landscape.tflite": "image_segmenter/selfie_segmenter_landscape/float16/{v}/selfie_segmenter_landscape.tflite",
}


# Optional face-swap models (InsightFace; non-commercial research license), name -> sha256.
FACESWAP_BASE = "https://huggingface.co/facefusion/models-3.0.0/resolve/main"
FACESWAP = {
    "arcface_w600k_r50.onnx": "f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1",
    "inswapper_128_fp16.onnx": "c4eccca86ad177586c85c28bf1a64a9d9ed237e283a15818d831f7facfd3f420",
}


# Optional meme sharpening (Real-ESRGAN, BSD-3), name -> sha256 (None: size check only).
ENHANCE = {"real_esrgan_x4_fp16.onnx": None}


def model_path(name: str) -> Path:
    return paths.models_dir() / name


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
    tmp.replace(dest)


def fetch(force: bool = False, quiet: bool = False) -> dict[str, str]:
    """Download missing models (pinned version 1, falling back to latest); return name -> sha256."""
    paths.models_dir().mkdir(parents=True, exist_ok=True)
    lock: dict[str, str] = {}
    for name, rel in MODELS.items():
        dest = model_path(name)
        if force or not dest.exists():
            for version in ("1", "latest"):
                url = f"{BASE}/{rel.format(v=version)}"
                try:
                    if not quiet:
                        print(f"downloading {name} ({version}) ...")
                    _download(url, dest)
                    break
                except urllib.error.HTTPError as e:
                    if e.code != 404 or version == "latest":
                        raise
        lock[name] = hashlib.sha256(dest.read_bytes()).hexdigest()
        if not quiet:
            print(f"  {name}: {dest.stat().st_size / 1e6:.1f} MB  sha256 {lock[name][:12]}")
    (paths.models_dir() / "models.lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return lock


def missing() -> list[str]:
    return [n for n in MODELS if not model_path(n).exists()]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch_faceswap(force: bool = False, quiet: bool = False) -> None:
    """Download and verify the face-swap models (your face on face-swap memes such as Harvey)."""
    paths.models_dir().mkdir(parents=True, exist_ok=True)
    if not quiet:
        print("Face-swap models: InsightFace inswapper + ArcFace, for non-commercial use only.")
    for name, sha in FACESWAP.items():
        dest = model_path(name)
        if force or not dest.exists():
            if not quiet:
                print(f"downloading {name} ...")
            _download(f"{FACESWAP_BASE}/{name}", dest)
        got = _sha256(dest)
        if got != sha:
            dest.unlink()
            raise RuntimeError(f"{name}: checksum mismatch (got {got[:12]}), deleted; try again")
        if not quiet:
            print(f"  {name}: {dest.stat().st_size / 1e6:.1f} MB  ok")


def fetch_enhance(force: bool = False, quiet: bool = False) -> None:
    """Download the meme sharpening model (small meme GIFs are AI-upscaled once, on the GPU)."""
    paths.models_dir().mkdir(parents=True, exist_ok=True)
    for name in ENHANCE:
        dest = model_path(name)
        if force or not dest.exists():
            if not quiet:
                print(f"downloading {name} ...")
            _download(f"{FACESWAP_BASE}/{name}", dest)
        if not quiet:
            print(f"  {name}: {dest.stat().st_size / 1e6:.1f} MB")
