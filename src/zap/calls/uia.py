"""UI Automation helpers shared by the call apps (Windows; pywinauto is imported on first use).

:func:`look` takes one snapshot of the visible top-level windows by program; the app modules search
those for their call window and its buttons. Two rules are enforced here for every app: Enter is
never pressed (whatever has the default focus is not ours to choose), and a button that ends the
call or meeting for everyone is never pressed (except Zoom's own end_for_all setting).
"""

from __future__ import annotations

import logging
import ntpath
import re
import time
import warnings
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

POLL_STEP_S = 0.1
WEB_WAIT_S = 3.0  # web content (Teams, WhatsApp, Discord, Signal...) builds its UIA tree only once asked: retry
MAX_NODES = 4000  # a tree walk gives up after this many elements

# "End for all / everyone" in the languages seen in Zoom, Teams, Meet, Webex, Jitsi and Telegram.
FORBIDDEN_RE = re.compile(
    r"(?i)for (all|everyone)|für alle|para todos|pour tous|per tutti|voor iedereen|dla wszystkich|для всех|"
    r"för alla|herkes için|untuk semua|cho tất cả|全員|모두에|^结束会议$|^結束會議$|^end meeting$|^end video chat$"
)


@dataclass
class Env:
    """pywinauto's entry points and one look at the visible top-level windows: (exe, pid, UIA window)."""

    Desktop: Any
    process_module: Callable[[int], str]
    send_keys: Callable[[str], None]
    windows: list[tuple[str, int, Any]] = field(default_factory=list)
    _exes: dict[int, str] = field(default_factory=dict)

    def exe(self, pid: int) -> str:
        """Lower-case program file name of a process ("" if it's gone or not ours to see)."""
        if pid not in self._exes:
            try:
                self._exes[pid] = ntpath.basename(self.process_module(pid) or "").lower()  # a Windows path, on any OS
            except Exception:  # noqa: BLE001 - process gone or access denied
                self._exes[pid] = ""
        return self._exes[pid]

    def of(self, exes: Iterable[str]) -> list[tuple[str, int, Any]]:
        wanted = set(exes)
        return [w for w in self.windows if w[0] in wanted]

    def fresh(self, exes: Iterable[str]) -> list[tuple[str, int, Any]]:
        """Like :meth:`of`, but looking again: for dialogs that open after a key or a click."""
        wanted = set(exes)
        return [w for w in _top_windows(self) if w[0] in wanted]

    def send(self, keys: str) -> None:
        send(self.send_keys, keys)


def api() -> tuple[Any, Any, Any]:
    """(Desktop, process_module, send_keys) from pywinauto, imported on first use."""
    with warnings.catch_warnings():  # "Revert to STA COM threading mode" is harmless here
        warnings.simplefilter("ignore", UserWarning)
        from pywinauto import Desktop, application, keyboard

    return Desktop, application.process_module, keyboard.send_keys


def look() -> Env:
    """Snapshot of the visible top-level windows and the programs they belong to."""
    Desktop, process_module, send_keys = api()
    env = Env(Desktop, process_module, send_keys)
    env.windows = _top_windows(env)
    return env


def _top_windows(env: Env) -> list[tuple[str, int, Any]]:
    out = []
    for w in env.Desktop(backend="uia").windows(visible_only=True):
        pid = _attr(w, "process_id")
        if pid is not None:  # None: closed while we looked
            out.append((env.exe(pid), pid, w))
    return out


def _attr(elem: Any, method: str) -> Any:
    """``elem.<method>()``, or None if the element vanished meanwhile."""
    try:
        return getattr(elem, method)()
    except Exception:  # noqa: BLE001 - element vanished
        return None


def send(send_keys: Callable[[str], None], keys: str) -> None:
    if "ENTER" in keys.upper() or "~" in keys:  # "~" is Enter in pywinauto's key syntax
        raise ValueError("ZAP never presses Enter in a call app")
    send_keys(keys)


def text(elem: Any) -> str:
    try:
        return (elem.window_text() or "").strip()
    except Exception:  # noqa: BLE001 - element vanished
        return ""


def auto_id(elem: Any) -> str:
    try:
        return elem.automation_id() or ""
    except Exception:  # noqa: BLE001 - element vanished, or not a UIA element
        return ""


def buttons(win: Any) -> list[Any]:
    try:
        return list(win.descendants(control_type="Button"))
    except Exception:
        log.debug("skipping a window (closed while we looked)", exc_info=True)
        return []


def find_button(win: Any, names: re.Pattern | None = None, ids: tuple[str, ...] = ()) -> Any | None:
    """The first button whose name matches ``names`` or whose AutomationId is in ``ids``."""
    for b in buttons(win):
        if (ids and auto_id(b) in ids) or (names is not None and names.search(text(b))):
            return b
    return None


def press(button: Any, allow_end_for_all: bool = False) -> None:
    """Invoke a button, refusing any that ends the call for everyone."""
    name = text(button)
    if FORBIDDEN_RE.search(name) and not allow_end_for_all:
        raise ValueError(f"ZAP never presses '{name}'")
    log.info("pressing '%s'", name or auto_id(button))
    button.invoke()


def wait_for(fn: Callable[[], Any], timeout: float) -> Any:
    """Call ``fn`` until it returns something truthy or ``timeout`` runs out; its last result."""
    deadline = time.perf_counter() + timeout
    while True:
        out = fn()
        if out or time.perf_counter() >= deadline:
            return out
        time.sleep(POLL_STEP_S)


def walk(root: Any, control_types: tuple[str, ...], skip: tuple[str, ...] = ("Document",)) -> Iterator[Any]:
    """Elements of the given control types under ``root``, without entering ``skip`` types (web pages)."""
    queue, seen = deque([root]), 0
    while queue and seen < MAX_NODES:
        for kid in _attr(queue.popleft(), "children") or ():
            seen += 1
            kind = _kind(kid)
            if kind in control_types:
                yield kid
            if kind is not None and kind not in skip:
                queue.append(kid)


def _kind(elem: Any) -> str | None:
    try:
        return elem.element_info.control_type
    except Exception:  # noqa: BLE001 - element vanished
        return None
