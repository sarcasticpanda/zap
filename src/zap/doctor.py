"""``zap doctor``: check this PC is ready for ZAP and say how to fix what is not.

Most ZAP problems are environmental: a second OpenCV wheel shadowing the
contrib build, Smart App Control blocking MediaPipe's unsigned DLL, OBS's
virtual camera missing or busy, Zoom's own gesture reactions fighting ours.
Each check prints one OK/WARN/FAIL line with a fix, and the exit code is 1 if
anything FAILs. Nothing here opens the camera or microphone, so it is safe to
run while ZAP or Zoom is using them, and it finishes in a few seconds.
"""

# ruff: noqa: BLE001  (every failure becomes a checklist line instead of a crash)
from __future__ import annotations

import contextlib
import ctypes
import importlib.metadata as md
import os
import sys
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass

from zap import paths

OBS_VCAM_CLSID = "{A3FCE0F5-3493-419F-958A-ABA1250EC20B}"
CV2_PROVIDERS = (
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
)
MEME_KEYS = ("michael_scott_forehead", "jim_stare", "harvey_car", "crying_jordan")

ZOOM_CHECKLIST = (
    "Video > Camera: choose 'OBS Virtual Camera' (never your physical webcam).",
    "Video: turn ON 'HD'.",
    "Video: untick 'Mirror my video' so captions read correctly (others never see it mirrored).",
    "Video > turn OFF: Touch up my appearance, Adjust for low light, Portrait lighting, auto-framing.",
    (
        "Background & effects: choose 'None' (no blur, no virtual background), no video filters, "
        "no Studio Effects, no avatars."
    ),
    (
        "Settings > Reactions: turn OFF 'Use hand gestures to activate reactions' "
        "(and in the meeting: React menu > untick hand gestures)."
    ),
    "Start ZAP before joining the meeting, so Zoom finds the virtual camera running.",
)


@dataclass
class Check:
    status: str  # OK, WARN, FAIL or INFO
    title: str
    hint: str = ""


# ---------------------------------------------------------------- helpers
@contextlib.contextmanager
def _quiet_native_stderr():
    """Hide MediaPipe's C++ log lines (written straight to fd 2) while models load."""
    try:
        sys.stderr.flush()
        saved = os.dup(2)
    except (OSError, ValueError):
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def _reg_value(root, path: str, name: str | None):
    """A registry value, True for an existing key when `name` is None, or None."""
    try:
        import winreg

        with winreg.OpenKey(root, path) as k:
            return True if name is None else winreg.QueryValueEx(k, name)[0]
    except OSError:
        return None


def _process_names() -> set[str]:
    """Lower-case exe names of running processes (Toolhelp snapshot; no extra packages)."""

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if not snap or snap == wintypes.HANDLE(-1).value:
        return set()
    names: set[str] = set()
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        ok = k32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            names.add(entry.szExeFile.lower())
            ok = k32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snap)
    return names


def cv2_provider_check(installed: list[str]) -> Check:
    """`installed`: distribution names (any case) that provide cv2."""
    names = sorted({n.lower() for n in installed})
    if names == ["opencv-contrib-python"]:
        return Check("OK", "OpenCV: only opencv-contrib-python is installed")
    if not names:
        return Check("FAIL", "OpenCV is not installed", "reinstall ZAP (uv sync)")
    others = [n for n in names if n != "opencv-contrib-python"]
    return Check(
        "FAIL",
        f"OpenCV: conflicting packages installed: {', '.join(names)}",
        f"uninstall {' '.join(others)}, then reinstall opencv-contrib-python (they overwrite each other's cv2)",
    )


# ---------------------------------------------------------------- checks
def check_python() -> Check:
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 12):
        return Check("OK", f"Python {ver}")
    return Check("FAIL", f"Python {ver} is too old", "install Python 3.12 or newer")


def check_cv2() -> list[Check]:
    installed = [d.metadata["Name"] for d in md.distributions() if (d.metadata["Name"] or "").lower() in CV2_PROVIDERS]
    out = [cv2_provider_check(installed)]
    try:
        import cv2
    except Exception as e:
        return out + [Check("FAIL", f"cannot import cv2: {e}", "reinstall opencv-contrib-python")]
    gui = "Win32 UI" in cv2.getBuildInformation()
    out.append(Check("OK", f"OpenCV {cv2.__version__} has a GUI") if gui else Check(
        "FAIL", f"OpenCV {cv2.__version__} has no GUI (headless build)", "uninstall opencv-*-headless"))
    out.append(Check("OK", "OpenCV contrib (ximgproc) available") if hasattr(cv2, "ximgproc") else Check(
        "FAIL", "OpenCV contrib modules missing", "install opencv-contrib-python instead of opencv-python"))
    return out


def check_mediapipe() -> list[Check]:
    os.environ.setdefault("GLOG_minloglevel", "2")
    from zap import fetch_models

    try:
        with _quiet_native_stderr():
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
    except Exception as e:
        return [Check("FAIL", f"cannot import MediaPipe: {e}",
                      "if this mentions a DLL, Smart App Control may be blocking it (see below); else reinstall")]
    out = [Check("OK", f"MediaPipe {mp.__version__}")]
    missing = set(fetch_models.missing())
    makers = {
        "gesture_recognizer.task": lambda b: vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(base_options=BaseOptions(model_asset_buffer=b))),
        "face_landmarker.task": lambda b: vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_buffer=b))),
        "selfie_segmenter_landscape.tflite": lambda b: vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_buffer=b), output_confidence_masks=True)),
    }
    for name, make in makers.items():
        if name in missing:
            out.append(Check("FAIL", f"model missing: {name}", "run: zap fetch-models"))
            continue
        try:
            with _quiet_native_stderr():
                make(fetch_models.model_path(name).read_bytes()).close()
            out.append(Check("OK", f"model loads: {name}"))
        except Exception as e:
            out.append(Check("FAIL", f"model does not load: {name} ({e})", "run: zap fetch-models --force"))
    return out


def check_cameras() -> tuple[list[Check], list[str]]:
    from zap.devices.camera import CameraError, list_cameras, pick_camera

    try:
        cams = list_cameras()
    except Exception as e:
        return [Check("FAIL", f"cannot list cameras: {e}", "reinstall cv2-enumerate-cameras")], []
    names = [c.name for c in cams]
    listing = ", ".join(f"[{c.index}] {c.name}{' (virtual)' if c.virtual else ''}" for c in cams) or "none"
    out = [Check("INFO", f"cameras: {listing}")]
    try:
        cam = pick_camera("", cams)
        out.append(Check("OK", f"ZAP will capture from: [{cam.index}] {cam.name}"))
    except CameraError as e:
        out.append(Check("FAIL", str(e), "plug in a webcam, or set camera.name in config.toml"))
    return out, names


def check_obs(camera_names: list[str]) -> list[Check]:
    import winreg

    listed = any("obs virtual camera" in n.lower() for n in camera_names)
    registered = _reg_value(winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\Classes\CLSID\{OBS_VCAM_CLSID}", None)
    if listed or registered:
        out = [Check("OK", "OBS Virtual Camera is installed")]
    else:
        out = [Check("FAIL", "OBS Virtual Camera is not installed",
                     "install OBS Studio (free); it registers the virtual camera ZAP sends to")]
    if "obs64.exe" in _process_names():
        out.append(Check("WARN", "OBS Studio is running",
                         "if OBS's 'Start Virtual Camera' is on, stop it (or close OBS): only one app can feed it"))
    return out


def check_audio() -> list[Check]:
    try:
        import sounddevice as sd

        from zap.devices.audio_out import CABLE_NAME, find_output_device
    except Exception as e:
        return [Check("WARN", f"audio unavailable: {e}", "snap detection needs a microphone; reinstall sounddevice")]
    apis = sd.query_hostapis()
    wasapi = next((i for i, a in enumerate(apis) if "wasapi" in a["name"].lower()), None)
    mics = [d for d in sd.query_devices() if wasapi is not None and d["hostapi"] == wasapi and d["max_input_channels"] > 0]
    if mics:
        default = apis[wasapi]["default_input_device"]
        name = sd.query_devices(default)["name"] if default is not None and default >= 0 else mics[0]["name"]
        out = [Check("OK", f"microphone (WASAPI): {name}")]
    else:
        out = [Check("WARN", "no WASAPI microphone found",
                     "plug in or enable a microphone; without one only single snaps (seen by the camera) work")]
    if find_output_device(CABLE_NAME) is not None:
        out.append(Check("INFO", "VB-CABLE found: sound.output = \"meeting\" can play effects into Zoom"))
    else:
        out.append(Check("INFO", "VB-CABLE not found: sound effects play on your speakers only",
                         "optional: install VB-CABLE and set sound.output = \"meeting\" so the meeting hears them"))
    return out


def check_smart_app_control() -> Check:
    import winreg

    state = _reg_value(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\CI\Policy",
                       "VerifiedAndReputablePolicyState")
    if state == 1:
        return Check("FAIL", "Smart App Control is ON",
                     "it blocks MediaPipe's unsigned DLL, so hand/face tracking cannot load. Turn it off in "
                     "Windows Security > App & browser control > Smart App Control (this cannot be undone)")
    if state == 2:
        return Check("WARN", "Smart App Control is in evaluation mode",
                     "Windows may switch it on and then block MediaPipe's unsigned DLL")
    return Check("OK", "Smart App Control is off")


def check_assets() -> list[Check]:
    from zap.assets import CLIP_EXT, IMAGE_EXT, SOUND_EXT
    from zap.render.synth_sfx import KEYS as SFX_KEYS

    def present(folder, key, exts) -> bool:
        return any((folder / f"{key}{e}").is_file() for e in exts)

    memes, sfx = paths.memes_dir(), paths.sfx_dir()
    out = [Check("INFO", f"ZAP folder: {paths.home()}")]
    have = [k for k in MEME_KEYS if present(memes, k, IMAGE_EXT + CLIP_EXT)]
    lack = [k for k in MEME_KEYS if k not in have]
    if lack:
        out.append(Check("INFO", f"memes: {len(have)}/{len(MEME_KEYS)} present; missing {', '.join(lack)} "
                         "(procedural stand-ins are used)", f"add your own images, e.g. {memes / (lack[0] + '.png')}"))
    else:
        out.append(Check("OK", f"memes: all {len(MEME_KEYS)} present"))
    have_sfx = [k for k in SFX_KEYS if present(sfx, k, SOUND_EXT)]
    out.append(Check("INFO", f"sounds: {len(have_sfx)}/{len(SFX_KEYS)} custom WAVs "
                     f"({', '.join(have_sfx) or 'none'}); the rest are synthesized",
                     f"optional: add <key>.wav to {sfx} for {', '.join(SFX_KEYS)}"))
    return out


def _safe(fn: Callable, *args) -> list[Check]:
    try:
        r = fn(*args)
    except Exception as e:  # a broken check must not hide the others
        return [Check("WARN", f"{getattr(fn, '__name__', 'check')} failed: {e}")]
    return r if isinstance(r, list) else [r]


def collect() -> list[Check]:
    checks = _safe(check_python) + _safe(check_cv2) + _safe(check_mediapipe)
    try:
        cam_checks, names = check_cameras()
    except Exception as e:
        cam_checks, names = [Check("WARN", f"camera check failed: {e}")], []
    checks += cam_checks + _safe(check_obs, names) + _safe(check_audio) + _safe(check_smart_app_control)
    return checks + _safe(check_assets)


def format_check(c: Check) -> str:
    line = f"[{c.status:^4}] {c.title}"
    return line + (f"\n       fix: {c.hint}" if c.hint and c.status != "OK" else "")


def run(zoom: bool = False, out: Callable[[str], None] = print) -> int:
    """Print the checklist; return 1 if anything FAILed, else 0."""
    out("ZAP doctor")
    checks = collect()
    for c in checks:
        out(format_check(c))
    out("")
    out("Zoom settings checklist:")
    for i, item in enumerate(ZOOM_CHECKLIST, 1):
        out(f"  {i}. {item}")
    if zoom:
        out("")
        out("Zoom windows (for leaving the meeting by gesture):")
        try:
            from zap.devices import zoom_ctl

            out(zoom_ctl.format_probe(zoom_ctl.probe()))
        except Exception as e:
            out(f"  could not inspect Zoom: {e}")
    fails = sum(c.status == "FAIL" for c in checks)
    warns = sum(c.status == "WARN" for c in checks)
    out("")
    out(f"{fails} problem(s), {warns} warning(s)." if fails or warns else "All good.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(run(zoom="--zoom" in sys.argv[1:]))
