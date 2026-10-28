"""Where ZAP keeps its files.

Everything user-specific lives in one per-user folder (``%APPDATA%\\ZAP`` on
Windows) so an installed build works on anyone's machine. Set the ``ZAP_HOME``
environment variable to use a different folder, e.g. while developing.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def home() -> Path:
    override = os.environ.get("ZAP_HOME")
    if override:
        return Path(override)
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home() / ".config"
    return base / "ZAP"


def models_dir() -> Path:
    return home() / "models"


def memes_dir() -> Path:
    return home() / "assets" / "memes"


def sfx_dir() -> Path:
    return home() / "assets" / "sfx"


def config_path() -> Path:
    return home() / "config.toml"


def calibration_dir() -> Path:
    return home() / "calibration"


def recordings_dir() -> Path:
    return home() / "recordings"


def logs_dir() -> Path:
    return home() / "logs"


def ensure_dirs() -> None:
    for d in (models_dir(), memes_dir(), sfx_dir(), calibration_dir(), recordings_dir(), logs_dir()):
        d.mkdir(parents=True, exist_ok=True)


def font_path(name: str) -> Path | None:
    """Resolve a font file name against the Windows font folder."""
    p = Path(name)
    if p.is_file():
        return p
    windir = os.environ.get("WINDIR", r"C:\Windows")
    candidate = Path(windir) / "Fonts" / name
    return candidate if candidate.is_file() else None
