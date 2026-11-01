"""What every call app shares: a found call, the result of leaving it, and the leave options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LeaveResult:
    ok: bool
    message: str
    dry_run: bool = False
    app: str = ""  # display name of the app that was left (or tried), "" when none was found


@dataclass
class Call:
    """A call in progress, as found on screen."""

    app: str  # key in APPS: "zoom", "teams", "meet", ...
    window: Any  # the window (pywinauto wrapper) that holds it
    pid: int = 0
    where: str = "app"  # "app" for a desktop app, else the browser it runs in ("Chrome", "Edge", ...)
    handle: Any = None  # whatever the app needs to leave it again: the button, the browser tab ...


@dataclass(frozen=True)
class Options:
    end_for_all: bool = False  # Zoom hosts: end the meeting for everyone instead of leaving
    discord_keys: str = ""  # pywinauto keys of the user's Discord "Disconnect" keybind ("" = none bound)


# Every app ZAP can leave, by config key (call.app); the browser sites share the keys of their desktop apps.
APPS: dict[str, str] = {
    "zoom": "Zoom",
    "teams": "Microsoft Teams",
    "meet": "Google Meet",
    "whatsapp": "WhatsApp",
    "discord": "Discord",
    "telegram": "Telegram",
    "signal": "Signal",
    "webex": "Webex",
    "slack": "Slack",
    "jitsi": "Jitsi Meet",
    "facetime": "FaceTime",
    "facebook": "Facebook Messenger",
    "browser": "a browser call",
}
APP_CHOICES = ("auto", *APPS)


def name_of(app: str) -> str:
    return "the call" if app == "auto" else APPS.get(app, app)
