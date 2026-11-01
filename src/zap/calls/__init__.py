"""Leave the call by gesture (snap twice), in whichever call app you are in.

ZAP leaves the way a person would: with the app's own hang-up shortcut where it has one, else its
Leave / End call button, found by UI Automation; a call in a browser tab is left with the site's
hang-up button or by closing that one tab. Per app, see :mod:`zap.calls.zoom`,
:mod:`zap.calls.apps` and :mod:`zap.calls.browser`. Everywhere: Enter is never pressed, "end for
everyone" is never chosen (except Zoom's call.end_for_all), and when no call or more than one is
found nothing is done and the user is told why.

Windows only for now: on macOS and Linux the countdown runs and the user hangs up themselves.
``leave`` blocks for up to a few seconds and never raises; call it off the frame loop.
"""

from __future__ import annotations

import logging
import sys

from zap.calls.base import APP_CHOICES, APPS, Call, LeaveResult, Options, name_of

__all__ = ["APPS", "APP_CHOICES", "Call", "LeaveResult", "Options", "keys_from_combo", "leave", "probe"]

log = logging.getLogger(__name__)

PLATFORM = sys.platform
_MODS = {"ctrl": "^", "control": "^", "shift": "+", "alt": "%"}


def leave(app: str = "auto", end_for_all: bool = False, dry_run: bool = True, discord_keys: str = "") -> LeaveResult:
    """Leave the call in ``app`` ("auto": the one app that is in a call). Never raises."""
    name = name_of(app)
    if dry_run:
        log.warning("WOULD LEAVE: dry run, not leaving %s (set call.dry_run = false to do it)", name)
        return LeaveResult(True, f"dry run: would leave {name}", dry_run=True)
    if PLATFORM != "win32":
        return LeaveResult(False, "Leaving by gesture works on Windows only for now: hang up yourself.")
    try:
        from zap.calls import windows

        return windows.leave(app, Options(end_for_all, keys_from_combo(discord_keys)))
    except Exception as e:
        log.exception("leaving the call failed")
        return LeaveResult(False, f"Could not leave {name}: {e}. Hang up yourself.")


def probe() -> str:
    """What ZAP sees of the open call apps, as text (``zap doctor --calls``)."""
    if PLATFORM != "win32":
        return "  (leaving by gesture works on Windows only for now)"
    from zap.calls import windows

    return windows.format_probe(windows.probe())


def keys_from_combo(combo: str) -> str:
    """A key combination as people write it ("ctrl+shift+f9") -> pywinauto keys ("^+{F9}")."""
    if not combo.strip():
        return ""
    *mods, key = [p.strip().lower() for p in combo.split("+")]
    bad = [m for m in mods if m not in _MODS]
    if bad or not key:
        raise ValueError(f"can't read the keys '{combo}' (write them like ctrl+shift+f9)")
    if key in ("enter", "return"):
        raise ValueError("ZAP never presses Enter in a call app")
    out = "".join(_MODS[m] for m in mods)
    return out + (key if len(key) == 1 and key.isalnum() else "{" + key.upper() + "}")
