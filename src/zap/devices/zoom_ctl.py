"""Leave the Zoom meeting from a gesture, by driving the Zoom desktop app's UI.

Zoom has no local API for this, so ZAP does what a person would: focus the
meeting window, press Alt+Q (Zoom's "leave" shortcut) and click "Leave meeting"
(or "End meeting for all" when configured) through UI Automation. Pressing
Enter is never done: whatever button has the default focus is not ours to
choose. If Zoom asks the host to assign a new host, or no button shows up, the
popup is dismissed with Esc and the user is told why.

pywinauto is imported lazily so this module loads without Zoom (or COM) and a
dry run never touches it. ``leave`` blocks for up to about 2.5 s; call it off
the frame loop.
"""

from __future__ import annotations

import logging
import os
import re
import time
import warnings
from dataclasses import dataclass

log = logging.getLogger(__name__)

MEETING_CLASSES = ("ConfMultiTabContentWndClass", "ZPContentViewWndClass")
ZOOM_EXE = "zoom.exe"
LEAVE_RE = re.compile(r"(?i)^leave meeting$")
END_ALL_RE = re.compile(r"(?i)end meeting for all")
ASSIGN_RE = re.compile(r"(?i)assign")
POLL_S = 1.5
AFTER_S = 1.0  # watch for the "assign a new host" dialog after clicking
POLL_STEP_S = 0.1
MAX_PROBE_BUTTONS = 80
_ASSIGN_MSG = (
    "You are the host, so Zoom asks you to assign a new host before leaving. "
    "Pick one in Zoom, or set zoom.end_for_all = true to end the meeting for everyone."
)


@dataclass
class LeaveResult:
    ok: bool
    message: str
    dry_run: bool = False


def _api():
    """(Desktop, process_module, send_keys) from pywinauto, imported on first use."""
    with warnings.catch_warnings():  # "Revert to STA COM threading mode" is harmless here
        warnings.simplefilter("ignore", UserWarning)
        from pywinauto import Desktop, application, keyboard

    return Desktop, application.process_module, keyboard.send_keys


def _send(send_keys, keys: str) -> None:
    if "ENTER" in keys.upper() or "~" in keys:  # "~" is Enter in pywinauto's key syntax
        raise ValueError("ZAP never presses Enter in Zoom")
    send_keys(keys)


def _is_zoom(pid: int, process_module, cache: dict[int, bool]) -> bool:
    if pid not in cache:
        try:
            cache[pid] = os.path.basename(process_module(pid) or "").lower() == ZOOM_EXE
        except Exception:  # noqa: BLE001 - process gone or access denied
            cache[pid] = False
    return cache[pid]


def find_meeting_window():
    """The visible Zoom meeting window (pywinauto win32 wrapper), or None."""
    Desktop, process_module, _ = _api()
    cache: dict[int, bool] = {}
    desktop = Desktop(backend="win32")
    for cls in MEETING_CLASSES:
        for w in desktop.windows(class_name=cls, visible_only=True):
            if _is_zoom(w.process_id(), process_module, cache):
                return w
    return None


def _buttons(Desktop, pid: int):
    """Button wrappers of every top-level UIA window of the Zoom process."""
    for win in Desktop(backend="uia").windows(process=pid):
        try:
            yield from win.descendants(control_type="Button")
        except Exception:
            log.debug("skipping a Zoom window (closed while we looked)", exc_info=True)
            continue


def _title(button) -> str:
    try:
        return (button.window_text() or "").strip()
    except Exception:  # noqa: BLE001 - button vanished
        return ""


def _scan(Desktop, pid: int, pattern: re.Pattern) -> tuple[object | None, bool]:
    """(matching button or None, whether an assign-new-host prompt is showing)."""
    found, assign = None, False
    for b in _buttons(Desktop, pid):
        t = _title(b)
        if found is None and pattern.search(t):
            found = b
        if ASSIGN_RE.search(t):
            assign = True
    return found, assign


def leave(end_for_all: bool = False, dry_run: bool = True) -> LeaveResult:
    """Leave (or end for all) the current Zoom meeting. Never raises."""
    action = "end the meeting for everyone" if end_for_all else "leave the meeting"
    if dry_run:
        log.warning("WOULD LEAVE: dry run, not trying to %s (set zoom.dry_run = false to do it)", action)
        return LeaveResult(True, f"dry run: would {action}", dry_run=True)
    try:
        return _leave(end_for_all, action)
    except Exception as e:
        log.exception("leaving the meeting failed")
        return LeaveResult(False, f"Could not {action}: {e}. Leave from Zoom yourself.")


def _leave(end_for_all: bool, action: str) -> LeaveResult:
    Desktop, _, send_keys = _api()
    win = find_meeting_window()
    if win is None:
        return LeaveResult(False, "No Zoom meeting window found (is the meeting open in the Zoom desktop app?)")
    pid = win.process_id()
    pattern = END_ALL_RE if end_for_all else LEAVE_RE
    win.set_focus()
    _send(send_keys, "%q")

    button, assign = _wait(Desktop, pid, pattern, POLL_S)
    if button is None or assign:
        _send(send_keys, "{ESC}")
        if assign:
            return LeaveResult(False, _ASSIGN_MSG)
        which = "End meeting for all" if end_for_all else "Leave meeting"
        return LeaveResult(False, f"Zoom did not show a '{which}' button; leave from Zoom yourself.")
    log.info("clicking '%s'", _title(button))
    button.invoke()

    # A host who leaves gets an "assign a new host" prompt; never pick one for them.
    deadline = time.perf_counter() + AFTER_S
    while time.perf_counter() < deadline:
        _, assign = _scan(Desktop, pid, ASSIGN_RE)
        if assign:
            _send(send_keys, "{ESC}")
            return LeaveResult(False, _ASSIGN_MSG)
        time.sleep(POLL_STEP_S)
    return LeaveResult(True, f"asked Zoom to {action}")



def _wait(Desktop, pid: int, pattern: re.Pattern, timeout: float) -> tuple[object | None, bool]:
    deadline = time.perf_counter() + timeout
    while True:
        button, assign = _scan(Desktop, pid, pattern)
        if button is not None or assign or time.perf_counter() >= deadline:
            return button, assign
        time.sleep(POLL_STEP_S)


def probe() -> list[dict]:
    """Visible Zoom windows with their class, title and button titles (for ``zap doctor --zoom``)."""
    Desktop, process_module, _ = _api()
    cache: dict[int, bool] = {}
    pids: list[int] = []
    for w in Desktop(backend="win32").windows(visible_only=True):
        try:
            pid = w.process_id()
        except Exception:
            log.debug("skipping a window", exc_info=True)
            continue
        if pid not in pids and _is_zoom(pid, process_module, cache):
            pids.append(pid)
    rows: list[dict] = []
    for pid in pids:
        for win in Desktop(backend="uia").windows(process=pid, visible_only=True):
            try:
                cls, title = win.class_name(), win.window_text()
                buttons = [t for t in (_title(b) for b in win.descendants(control_type="Button")) if t]
            except Exception as e:  # noqa: BLE001 - report it in the probe
                rows.append({"pid": pid, "class": "?", "title": f"<error: {e}>", "meeting": False, "buttons": []})
                continue
            rows.append({
                "pid": pid,
                "class": cls,
                "title": title,
                "meeting": cls in MEETING_CLASSES,
                "buttons": buttons[:MAX_PROBE_BUTTONS],
            })
    return rows


def format_probe(rows: list[dict]) -> str:
    if not rows:
        return "  no visible Zoom windows (start Zoom and join a meeting, then run this again)"
    lines = []
    for r in rows:
        tag = "  [meeting window]" if r["meeting"] else ""
        lines.append(f"  pid {r['pid']}  class={r['class']!r}  title={r['title']!r}{tag}")
        if r["buttons"]:
            lines.append("    buttons: " + ", ".join(repr(b) for b in r["buttons"]))
    return "\n".join(lines)

