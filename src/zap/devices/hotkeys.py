"""Global keyboard shortcuts, active while Zoom has focus.

Every shortcut is Ctrl+Alt+key: Zoom binds plain Alt+letter (Alt+Q leaves the
meeting, Alt+A mutes), so a bare Alt combo could fire both apps. pynput calls
back on its own hook thread; commands are only queued there and the frame loop
applies them with :meth:`Hotkeys.drain`, so no app state is touched off-thread.
Hotkeys are a convenience: if the keyboard hook cannot start, ZAP keeps running.
"""

from __future__ import annotations

import logging
import queue
from collections.abc import Callable, Mapping
from typing import Self

log = logging.getLogger(__name__)

# pynput combo -> command. "trigger:<gesture>" fires a gesture as if it was performed.
BINDINGS: dict[str, str] = {
    "<ctrl>+<alt>+a": "arm_toggle",
    "<ctrl>+<alt>+p": "panic",
    "<ctrl>+<alt>+s": "sound_toggle",
    "<ctrl>+<alt>+b": "plate",
    "<ctrl>+<alt>+r": "reload",
    "<ctrl>+<alt>+c": "calibrate",
    "<ctrl>+<alt>+y": "confirm",
    "<ctrl>+<alt>+<up>": "sens_up",
    "<ctrl>+<alt>+<down>": "sens_down",
    "<ctrl>+<alt>+<end>": "trigger:snap2",
    "<ctrl>+<alt>+1": "trigger:snap1",
    "<ctrl>+<alt>+2": "trigger:two_thumbs_up",
    "<ctrl>+<alt>+3": "trigger:hands_up",
    "<ctrl>+<alt>+4": "trigger:cover_face",
    "<ctrl>+<alt>+5": "trigger:forehead",
    "<ctrl>+<alt>+6": "trigger:eyebrows_up",
    "<ctrl>+<alt>+7": "trigger:collar",
    "<ctrl>+<alt>+8": "trigger:peace",
    "<ctrl>+<alt>+9": "trigger:l_shape",
}

HOTKEY_HELP: list[tuple[str, str]] = [
    ("Ctrl+Alt+A", "arm / disarm gestures"),
    ("Ctrl+Alt+P", "panic: stop every effect and show the plain camera"),
    ("Ctrl+Alt+S", "sound effects on / off"),
    ("Ctrl+Alt+B", "re-capture the empty background plate"),
    ("Ctrl+Alt+R", "reload config and meme assets"),
    ("Ctrl+Alt+C", "calibrate every gesture to you (about a minute, follow the preview)"),
    ("Ctrl+Alt+Y", "confirm (same as a thumbs-up)"),
    ("Ctrl+Alt+Up / Down", "snap sensitivity up / down"),
    ("Ctrl+Alt+End", "vanish, then leave the meeting (snap x2)"),
    ("Ctrl+Alt+1", "Thanos vanish / reappear (snap x1)"),
    ("Ctrl+Alt+2", "money rain, DEAL CLOSED (thumbs up)"),
    ("Ctrl+Alt+3", "Absolute Cinema (both hands up)"),
    ("Ctrl+Alt+4", "crying Jordan + sad mode (cover face)"),
    ("Ctrl+Alt+5", "Michael Scott (hand on forehead)"),
    ("Ctrl+Alt+6", "Jim stare (eyebrows up)"),
    ("Ctrl+Alt+7", "Harvey Specter (fingers at your collar)"),
    ("Ctrl+Alt+8", "We'll Be Right Back (peace sign); again to come back"),
    ("Ctrl+Alt+9", "To Be Continued (L sign); again to come back"),
]


def _make_listener(hotkeys: Mapping[str, Callable[[], None]]):
    """GlobalHotKeys that also treats AltGr (right Alt, sent as Ctrl+AltGr) as Alt."""
    from pynput import keyboard

    class _Listener(keyboard.GlobalHotKeys):
        def canonical(self, key):
            if key == keyboard.Key.alt_gr:
                return keyboard.Key.alt
            return super().canonical(key)

    return _Listener(dict(hotkeys))


class Hotkeys:
    """Queue of hotkey commands; call :meth:`drain` once per frame."""

    def __init__(self, bindings: Mapping[str, str] | None = None) -> None:
        self.bindings = dict(BINDINGS if bindings is None else bindings)
        self.error = ""
        self._queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._listener = None

    @property
    def running(self) -> bool:
        return self._listener is not None and self._listener.is_alive()

    def push(self, command: str) -> None:
        """Queue a command (also how tests and the HUD inject commands)."""
        self._queue.put(command)

    def drain(self) -> list[str]:
        out: list[str] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                return out

    def start(self) -> bool:
        if self.running:
            return True
        try:
            callbacks = {combo: self._callback(cmd) for combo, cmd in self.bindings.items()}
            listener = _make_listener(callbacks)
            listener.daemon = True
            listener.start()
            listener.wait()
        except Exception as e:  # noqa: BLE001 - no keyboard hook (locked-down PC, remote session, ...)
            self.error = f"hotkeys unavailable: {e}"
            log.warning("%s; continuing without hotkeys", self.error)
            self._listener = None
            return False
        self._listener = listener
        self.error = ""
        log.info("hotkeys active (%d bindings)", len(self.bindings))
        return True

    def stop(self) -> None:
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                log.exception("error stopping hotkeys")

    def _callback(self, command: str) -> Callable[[], None]:
        def fire() -> None:
            self._queue.put(command)

        return fire

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
