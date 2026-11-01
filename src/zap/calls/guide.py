"""Setting up each call app for ZAP: a few short steps per app, for ``zap doctor`` and the README.

What the steps are based on: which camera system each app reads (DirectShow virtual cameras such
as OBS's reach Zoom, Teams, Discord, Telegram, Webex and everything Chromium-based: Meet and other
browser calls, Slack, Signal, the 2025+ WhatsApp app), each app's own effects that would cut into
ZAP's picture or fire on ZAP's gestures, and how ZAP leaves its calls (:mod:`zap.calls`).

Steps marked WIN only apply on Windows (leaving by gesture, DirectShow, VB-CABLE) and steps marked
ELSEWHERE only on macOS and Linux; :func:`steps` picks them for an OS and names the virtual camera
the way call apps list it there.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

WIN = "[win] "
ELSEWHERE = "[mac/linux] "
MAC_APPS = Path("/Applications")

EVERY_APP = (
    "Start ZAP before the call app turns your camera on (picture black? turn the camera off and on in the call).",
    "Camera: choose 'OBS Virtual Camera' (never your real webcam).",
    "Turn off the app's background blur/replace, filters and auto-framing: they cut into ZAP's memes.",
    "Your own preview may look mirrored; everyone else sees ZAP the right way round.",
    (WIN + "Meme sounds into the call: install VB-CABLE, set sound.output = \"meeting\" and pick 'CABLE Output' as "
     "the call's microphone; then turn the app's noise suppression down or it removes the music."),
    ELSEWHERE + "Snap twice makes you vanish; then hang up yourself (leaving by gesture is Windows-only for now).",
)

SETUP: dict[str, tuple[str, ...]] = {
    "zoom": (
        ("Settings > Reactions: turn OFF 'Use hand gestures to activate reactions' (it reacts to thumbs up, "
         "peace and other signs ZAP uses)."),
        "Video: turn off auto-framing, Touch up and Adjust for low light; Background & effects: None.",
        WIN + "Camera black or missing: Video > Advanced > Video capturing method: DirectShow, then restart Zoom.",
        WIN + "Snap twice leaves with Zoom's Alt+Q: keep that shortcut. Hosts: see call.end_for_all.",
    ),
    "teams": (
        ("Settings > Devices > Camera: OBS Virtual Camera. The preview must show you, not the OBS logo or a frozen "
         "picture; if it does, quit Teams from the tray and reopen it, or use Teams in Chrome or Edge."),
        WIN + "Never run ZAP as administrator (Teams then can't read its picture).",
        "Video effects: Background None, filters off.",
        WIN + "Snap twice leaves with Teams' Ctrl+Shift+H (it never ends the meeting for everyone).",
    ),
    "meet": (
        "In Chrome or Edge: Settings (gear) > Video > Camera: OBS Virtual Camera.",
        "Backgrounds and effects: 'Turn off visual effects', and Framing off.",
        "More options > Reactions: keep 'Hand raise gesture' off.",
        WIN + "Snap twice closes the Meet tab, which leaves the call.",
    ),
    "whatsapp": (
        ("In a video call, pick OBS Virtual Camera in the camera menu. If it isn't listed, call from "
         "web.whatsapp.com in Chrome or Edge instead."),
        WIN + "Snap twice presses WhatsApp's end-call button (or closes the WhatsApp Web tab).",
    ),
    "discord": (
        "User Settings > Voice & Video > Camera: OBS Virtual Camera; Video Background: None.",
        (WIN + "Snap twice to leave: Keybinds > add 'Disconnect from Voice Channel' (e.g. Ctrl+Shift+F9) and write "
         "the same keys in call.discord_keys (else ZAP presses the Disconnect button)."),
        "Meme sounds: Noise Suppression: None, or Discord removes them.",
    ),
    "telegram": (
        "Settings > Speakers and Camera > Camera: OBS Virtual Camera.",
        "Calls start as voice calls: press the camera button in the call.",
        WIN + "Snap twice presses 'End Call' (closing Telegram's call window doesn't hang up).",
    ),
    "signal": (
        "Settings > Calls > Camera: OBS Virtual Camera.",
        "Signal sends a 4:3 picture: memes near the far left and right may be cut off.",
        WIN + "Snap twice uses Signal's Alt+Shift+E.",
    ),
    "webex": (
        "Settings > Video > Camera: OBS Virtual Camera.",
        "In a meeting: Reactions > turn OFF 'Recognize hand gestures'.",
        (WIN + "Keep Settings > Meetings > 'Ask me to confirm when leaving' on: snap twice then picks 'Leave "
         "meeting', never 'End meeting for all'."),
    ),
    "slack": (
        "Huddle > More > Video and mic settings > Camera: OBS Virtual Camera; background: none.",
        WIN + "Snap twice leaves a huddle only with call.app = \"slack\".",
    ),
    "browser": (
        ("Any other call in Chrome, Edge or Firefox (Jitsi, Zoom or Discord on the web, FaceTime links, Facebook "
         "calls): pick OBS Virtual Camera in the site's camera menu or the browser's site settings."),
        WIN + "Snap twice closes that call's tab (it must be the only tab using your camera).",
    ),
}

TITLES = {"browser": "Other calls in a browser"}
NOT_SUPPORTED = (
    "Skype closed in May 2025: use Teams (free) instead.",
    WIN + "Viber doesn't list OBS Virtual Camera, so it can't show ZAP for now.",
)


def steps(lines: Iterable[str], platform: str = sys.platform) -> list[str]:
    """``lines`` for this OS: the other OS's steps dropped, and ZAP's camera by the name call apps show there."""
    from zap.devices.vcam import camera_name

    out = []
    for line in lines:
        for mark, here in ((WIN, platform == "win32"), (ELSEWHERE, platform != "win32")):
            if line.startswith(mark):
                line = line[len(mark):] if here else ""
        if line:
            out.append(line.replace("OBS Virtual Camera", camera_name(platform)))
    return out


def installed(platform: str = sys.platform) -> list[str]:
    """The call apps (SETUP keys) that look installed on this computer, and "browser"."""
    if platform == "darwin":
        names = {"zoom": "zoom.us", "teams": "Microsoft Teams", "whatsapp": "WhatsApp", "discord": "Discord",
                 "telegram": "Telegram", "signal": "Signal", "webex": "Webex", "slack": "Slack"}
        found = {k: (MAC_APPS / f"{v}.app").is_dir() for k, v in names.items()} | {"meet": True}  # Safari works too
    elif platform != "win32":
        def has(*commands: str) -> bool:
            return any(shutil.which(c) for c in commands)

        found = {"zoom": has("zoom"), "teams": has("teams-for-linux"), "discord": has("discord"),
                 "telegram": has("telegram-desktop", "Telegram"), "signal": has("signal-desktop"),
                 "slack": has("slack"), "meet": has("google-chrome", "chromium", "chromium-browser", "firefox",
                                                     "microsoft-edge")}
    else:
        found = _installed_windows()
    return [k for k in SETUP if found.get(k)] + ["browser"]


def _installed_windows() -> dict[str, bool]:
    local, roaming = Path(os.environ.get("LOCALAPPDATA", "")), Path(os.environ.get("APPDATA", ""))
    progs = [Path(os.environ.get(k, "")) for k in ("ProgramFiles", "ProgramFiles(x86)")]
    packages = local / "Packages"

    def package(prefix: str) -> bool:
        return packages.is_dir() and any(p.name.startswith(prefix) for p in packages.iterdir())

    return {
        "zoom": (roaming / "Zoom" / "bin" / "Zoom.exe").is_file() or any((p / "Zoom").is_dir() for p in progs),
        "teams": package("MSTeams_") or (local / "Microsoft" / "Teams").is_dir(),
        "meet": any((p / "Google" / "Chrome").is_dir() or (p / "Microsoft" / "Edge").is_dir() for p in progs),
        "whatsapp": package("5319275A.WhatsAppDesktop_") or (local / "WhatsApp").is_dir(),
        "discord": (local / "Discord").is_dir(),
        "telegram": package("TelegramMessengerLLP.") or (roaming / "Telegram Desktop").is_dir(),
        "signal": (local / "Programs" / "signal-desktop").is_dir(),
        "webex": (local / "CiscoSparkLauncher").is_dir() or (local / "Programs" / "Cisco Spark").is_dir(),
        "slack": (local / "slack").is_dir(),
    }
