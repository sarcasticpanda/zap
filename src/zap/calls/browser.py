"""Calls in a browser tab: Google Meet, Teams, Zoom, Discord, Jitsi, WhatsApp Web, FaceTime, Facebook...

The call tab is found in the browser's own tab strip (browser UI, not the page): Chrome and Edge add
" - Camera and microphone recording" (or "Camera recording", "Microphone recording") to the name of
a tab using the camera or mic, which marks a call whatever the site. Without that mark (Firefox, or
another language) only tabs of call-only sites (Meet, Jitsi, FaceTime) count. Exactly one call tab
must be found, or nothing is done.

Leaving: Teams, Discord and WhatsApp Web have a hang-up button the page exposes to UI Automation
(Teams by a language-independent id), pressed after the tab is selected. Everything else, and the
fallback, is closing that one tab with its own close button, which leaves the call in any language
without keystrokes (Meet: "the meeting ends when the last person leaves"). An installed web app
(PWA) window has no tabs: its window is closed instead.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from zap.calls import uia
from zap.calls.base import APPS, Call, LeaveResult, Options

log = logging.getLogger(__name__)

BROWSERS = {"chrome.exe": "Chrome", "msedge.exe": "Edge", "firefox.exe": "Firefox", "brave.exe": "Brave",
            "opera.exe": "Opera", "vivaldi.exe": "Vivaldi", "arc.exe": "Arc"}
RECORDING_RE = re.compile(r"(?i) - ((camera( and microphone)?|microphone) recording|sharing (your )?(screen|window|tab))")
SITES = (
    ("meet", re.compile(r"(?i)^meet\b|google meet")),
    ("teams", re.compile(r"(?i)microsoft teams")),
    ("zoom", re.compile(r"(?i)\bzoom\b")),
    ("discord", re.compile(r"(?i)\bdiscord\b")),
    ("jitsi", re.compile(r"(?i)jitsi")),
    ("whatsapp", re.compile(r"(?i)whatsapp")),
    ("facetime", re.compile(r"(?i)facetime")),
    ("facebook", re.compile(r"(?i)messenger|facebook")),
    ("webex", re.compile(r"(?i)webex")),
    ("slack", re.compile(r"(?i)\bslack\b")),
)
CLOSE_RE = re.compile(r"(?i)^(close|schließen|fermer|cerrar|chiudi|fechar|sluiten|zamknij|закрыть|stäng|kapat|"
                      r"tutup|đóng|閉じる|닫기|关闭|關閉)\b")  # the tab's close button (the last one otherwise)
CALL_ONLY = ("meet", "jitsi", "facetime")  # sites whose pages are calls: counted even without the recording mark
# in-page hang-up buttons worth pressing (name pattern, AutomationIds = HTML ids); the rest: close the tab
IN_PAGE = {
    "teams": (re.compile(r"(?i)^(leave|hang up)$"), ("hangup-button", "splitButton-ddd__primaryActionButton")),
    "discord": (re.compile(r"(?i)^disconnect$"), ()),
    "whatsapp": (re.compile(r"(?i)^(end( video)? call|leave call|hang ?up)$"), ()),
}


def site_of(title: str) -> str:
    return next((key for key, rx in SITES if rx.search(title)), "")


class Browsers:
    keys = tuple(k for k, _ in SITES) + ("browser",)
    exes = tuple(BROWSERS)
    lazy = False  # the tab strip is browser UI: found at once
    auto = True

    def find(self, env: uia.Env) -> list[Call]:
        out = []
        for exe, pid, win in env.of(self.exes):
            where = BROWSERS[exe]
            tabs = list(uia.walk(win, ("TabItem",)))
            if not tabs:  # an installed web app (PWA) window: no tab strip, its title is the page's
                site = site_of(uia.text(win))
                if site in CALL_ONLY:
                    out.append(Call(site, win, pid, where=f"{where} app"))
                continue
            for tab in tabs:
                name = uia.text(tab)
                site = site_of(RECORDING_RE.split(name)[0])
                if RECORDING_RE.search(name) or site in CALL_ONLY:
                    out.append(Call(site or "browser", win, pid, where=where, handle=tab))
        return out

    def leave(self, env: uia.Env, call: Call, opts: Options) -> LeaveResult:
        what = f"{APPS.get(call.app, 'the call')} in {call.where}"
        tab = call.handle
        if tab is None:  # PWA window
            call.window.close()
            return LeaveResult(True, f"closed {what}", app=APPS.get(call.app, "browser"))
        if call.app in IN_PAGE:
            _select(tab)
            names, ids = IN_PAGE[call.app]
            button = uia.wait_for(lambda: uia.find_button(call.window, names, ids), uia.WEB_WAIT_S)
            if button is not None:
                uia.press(button)
                return LeaveResult(True, f"left {what}", app=APPS[call.app])
            log.info("no hang-up button found in %s; closing the tab instead", what)
        return _close(env, call, what)


def _close(env: uia.Env, call: Call, what: str) -> LeaveResult:
    """Close exactly the call's tab: its own close button, else select it, check, and Ctrl+W."""
    tab = call.handle
    buttons = list(uia.walk(tab, ("Button",)))  # pinned or squeezed tabs show none; a playing tab adds "mute"
    close = next((b for b in buttons if CLOSE_RE.search(uia.text(b))), buttons[-1] if buttons else None)
    if close is not None:
        uia.press(close)
        return LeaveResult(True, f"closed the tab to leave {what}", app=APPS.get(call.app, "browser"))
    _select(tab)
    call.window.set_focus()
    if not _selected(tab):  # Ctrl+W closes whatever tab is active: only when it's surely ours
        return LeaveResult(False, f"couldn't select the tab of {what}; close it yourself.", app="browser")
    env.send("^w")
    return LeaveResult(True, f"closed the tab to leave {what}", app=APPS.get(call.app, "browser"))


def _select(tab: Any) -> None:
    try:
        tab.select()
    except Exception:
        log.debug("couldn't select a browser tab", exc_info=True)


def _selected(tab: Any) -> bool:
    try:
        return bool(tab.is_selected())
    except Exception:  # noqa: BLE001 - the tab vanished
        return False
