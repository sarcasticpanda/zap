"""Where ZAP keeps its files.

Everything user-specific lives in one per-user folder (``%APPDATA%\\ZAP`` on
Windows, ``~/Library/Application Support/ZAP`` on macOS, ``~/.config/ZAP`` on
Linux) so an installed build works on anyone's machine. Set the ``ZAP_HOME``
environment variable to use a different folder, e.g. while developing.
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

# The Windows fonts ZAP's effects name, and where else the same font (or a free look-alike) lives:
# macOS ships Microsoft's core fonts under their family names; Linux has them with msttcorefonts,
# else the Liberation/DejaVu fonts most distributions install.
FONT_ALIASES = {
    "timesbd.ttf": ("Times New Roman Bold.ttf", "Times_New_Roman_Bold.ttf", "LiberationSerif-Bold.ttf",
                    "Tinos-Bold.ttf", "DejaVuSerif-Bold.ttf"),
    "ariblk.ttf": ("Arial Black.ttf", "Arial_Black.ttf", "ArchivoBlack-Regular.ttf", "LiberationSans-Bold.ttf",
                   "DejaVuSans-Bold.ttf"),
    "georgiaz.ttf": ("Georgia Bold Italic.ttf", "Georgia_Bold_Italic.ttf", "Gelasio-BoldItalic.ttf",
                     "LiberationSerif-BoldItalic.ttf", "DejaVuSerif-BoldItalic.ttf"),
    "georgiai.ttf": ("Georgia Italic.ttf", "Georgia_Italic.ttf", "Gelasio-Italic.ttf", "LiberationSerif-Italic.ttf",
                     "DejaVuSerif-Italic.ttf"),
    "georgiab.ttf": ("Georgia Bold.ttf", "Georgia_Bold.ttf", "Gelasio-Bold.ttf", "LiberationSerif-Bold.ttf",
                     "DejaVuSerif-Bold.ttf"),
    "impact.ttf": ("Impact.ttf", "Anton-Regular.ttf", "LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf"),
    "arialbd.ttf": ("Arial Bold.ttf", "Arial_Bold.ttf", "LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf"),
    "arial.ttf": ("Arial.ttf", "LiberationSans-Regular.ttf", "DejaVuSans.ttf"),
    "segoeui.ttf": ("Helvetica.ttc", "LiberationSans-Regular.ttf", "DejaVuSans.ttf"),
    "seguiemj.ttf": ("NotoColorEmoji.ttf",),  # Apple Color Emoji has no 109 px size, which the emoji are drawn at
}


def home() -> Path:
    override = os.environ.get("ZAP_HOME")
    if override:
        return Path(override)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ZAP"
    appdata = os.environ.get("APPDATA") if sys.platform == "win32" else None
    base = Path(appdata) if appdata else Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
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
    """Resolve a font file name: a path, a file in the system's font folders, or its look-alike there."""
    p = Path(name)
    if p.is_file():
        return p
    fonts = _system_fonts()
    for candidate in (name, *FONT_ALIASES.get(name.lower(), ())):
        found = fonts.get(candidate.lower())
        if found is not None:
            return found
    return None


@functools.cache
def _system_fonts() -> dict[str, Path]:
    """Font file name (lower case) -> path, over this system's font folders."""
    if sys.platform == "win32":
        folders = [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
                   Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"]
    elif sys.platform == "darwin":
        folders = [Path("/System/Library/Fonts/Supplemental"), Path("/System/Library/Fonts"), Path("/Library/Fonts"),
                   Path.home() / "Library" / "Fonts"]
    else:
        folders = [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".local" / "share" / "fonts",
                   Path.home() / ".fonts"]
    out: dict[str, Path] = {}
    for folder in folders:
        if folder.is_dir():
            for f in folder.rglob("*.tt[fc]"):
                out.setdefault(f.name.lower(), f)
    return out
