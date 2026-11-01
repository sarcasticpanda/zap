"""Zoom desktop app: Alt+Q (Zoom's own "leave" shortcut), then the "Leave meeting" button.

Zoom has no local API for this, so ZAP does what a person would: focus the meeting window, press
Alt+Q and click "Leave meeting" (or "End meeting for all" when configured) through UI Automation,
in any of Zoom's languages (the texts come from Zoom's own language files). Webinar attendees get
"Leave webinar". If Zoom asks the host to assign a new host, or no button shows up, the popup is
dismissed with Esc and the user is told why.
"""

from __future__ import annotations

import logging
import re
import time

from zap.calls import uia
from zap.calls.base import Call, LeaveResult, Options

log = logging.getLogger(__name__)

EXES = ("zoom.exe",)
# the meeting window; ZPContentViewWndClass is the older Zoom's; while you share your screen the meeting
# window hides and only the floating toolbar is visible (Alt+Q from there: not verified on a real share)
MEETING_CLASSES = ("ConfMultiTabContentWndClass", "ZPContentViewWndClass", "ZPFloatToolbarClass")
LEAVE_RE = re.compile(
    r"(?i)^(leave (meeting|webinar)|meeting verlassen|webinar verlassen|salir|salir de la reunión|salir del seminario "
    r"web|quitter la réunion|quitter le webinaire|esci dalla riunione|sair da reunião|vergadering verlaten|opuść "
    r"spotkanie|выйти из конференции|lämna mötet|toplantıdan çık|keluar dari rapat|rời khỏi cuộc họp|"
    r"ミーティングを退出|회의 나가기|离开会议|離開會議)$"
)
END_ALL_RE = re.compile(
    r"(?i)end meeting for all|für alle beenden|para todos|pour tous|per tutti|voor iedereen|dla wszystkich|для всех|"
    r"för alla|herkes için|untuk semua|cho tất cả|全員に対して|모두에 대해|^结束会议$|^結束會議$"
)
ASSIGN_RE = re.compile(r"(?i)assign|zuweisen|asignar|attribuer|assegna|atribuir|toewijzen|przypisz|назнач|tilldela|"
                       r"割り当て|지정")
POLL_S = 1.5
AFTER_S = 1.0  # watch for the "assign a new host" dialog after clicking
_ASSIGN_MSG = (
    "You are the host, so Zoom asks you to assign a new host before leaving. "
    "Pick one in Zoom, or set call.end_for_all = true to end the meeting for everyone."
)


class Zoom:
    keys = ("zoom",)
    exes = EXES
    lazy = False  # a native window: found at once or not at all
    auto = True

    def find(self, env: uia.Env) -> list[Call]:
        win = meeting_window(env)
        return [] if win is None else [Call("zoom", win, win.process_id())]

    def leave(self, env: uia.Env, call: Call, opts: Options) -> LeaveResult:
        end_for_all = opts.end_for_all
        action = "end the meeting for everyone" if end_for_all else "leave the meeting"
        pattern = END_ALL_RE if end_for_all else LEAVE_RE
        call.window.set_focus()
        env.send("%q")

        button, assign = _wait(env, call.pid, pattern, POLL_S)
        if button is None or assign:
            env.send("{ESC}")
            if assign:
                return LeaveResult(False, _ASSIGN_MSG, app="Zoom")
            which = "End meeting for all" if end_for_all else "Leave meeting"
            return LeaveResult(False, f"Zoom did not show a '{which}' button; leave from Zoom yourself.", app="Zoom")
        uia.press(button, allow_end_for_all=end_for_all)

        # A host who leaves gets an "assign a new host" prompt; never pick one for them.
        deadline = time.perf_counter() + AFTER_S
        while time.perf_counter() < deadline:
            _, assign = _scan(env, call.pid, ASSIGN_RE)
            if assign:
                env.send("{ESC}")
                return LeaveResult(False, _ASSIGN_MSG, app="Zoom")
            time.sleep(uia.POLL_STEP_S)
        return LeaveResult(True, f"asked Zoom to {action}", app="Zoom")


def meeting_window(env: uia.Env):
    """The visible Zoom meeting window (pywinauto win32 wrapper), or None."""
    desktop = env.Desktop(backend="win32")
    for cls in MEETING_CLASSES:
        for w in desktop.windows(class_name=cls, visible_only=True):
            if env.exe(w.process_id()) in EXES:
                return w
    return None


def _buttons(env: uia.Env, pid: int):
    """Button wrappers of every top-level UIA window of the Zoom process."""
    for win in env.Desktop(backend="uia").windows(process=pid):
        yield from uia.buttons(win)


def _scan(env: uia.Env, pid: int, pattern: re.Pattern) -> tuple[object | None, bool]:
    """(matching button or None, whether an assign-new-host prompt is showing)."""
    found, assign = None, False
    for b in _buttons(env, pid):
        t = uia.text(b)
        if found is None and pattern.search(t):
            found = b
        if ASSIGN_RE.search(t):
            assign = True
    return found, assign


def _wait(env: uia.Env, pid: int, pattern: re.Pattern, timeout: float) -> tuple[object | None, bool]:
    deadline = time.perf_counter() + timeout
    while True:
        button, assign = _scan(env, pid, pattern)
        if button is not None or assign or time.perf_counter() >= deadline:
            return button, assign
        time.sleep(uia.POLL_STEP_S)
