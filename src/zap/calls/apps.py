"""Desktop call apps other than Zoom: Teams, Telegram, Signal, Discord, WhatsApp, Webex, Slack.

Each finds its call window by the app's own hang-up button, then leaves with the app's own
shortcut where it has one, checking that the call really ended, and presses the button otherwise:

- Teams (new Teams, also Teams free, which replaced Skype): Ctrl+Shift+H in the meeting window;
  the meeting is found by language-independent AutomationIds.
- Signal: Alt+Shift+E in the call window.
- Webex: Ctrl+L, then "Leave meeting" in the confirmation (never "End meeting for all").
- Discord: the user's own "Disconnect from Voice Channel" keybind if set (Discord has none by
  default), else the "Disconnect" button.
- Telegram: the "End Call" button ("Leave" in a group video chat, then "Leave" again to confirm);
  closing Telegram's call window only hides it.
- WhatsApp (the 2025+ app is WhatsApp Web in a WebView2 window): its end-call button.
- Slack: the huddle's "Leave" button, only when call.app = "slack" (its Ctrl+Shift+H would START a
  huddle when none is open, so it is never sent).

Button names other than Teams' AutomationIds are the English ones, which these apps show in
English; `zap doctor --calls` lists what an app really shows, to add more.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from zap.calls import uia
from zap.calls.base import Call, LeaveResult, Options

log = logging.getLogger(__name__)

GONE_S = 2.5  # after the shortcut, the call must be gone within this long, else its button is pressed


class _ButtonApp:
    """A call app whose call window is recognised by its hang-up button."""

    key = ""
    name = ""
    exes: tuple[str, ...] = ()
    hangup = re.compile(r"$^")  # the hang-up button's name
    hangup_ids: tuple[str, ...] = ()  # ... or AutomationId
    shortcut = ""  # pywinauto keys that hang up, sent to the focused call window ("" = press the button)
    lazy = True  # web content (WebView2 / Electron) builds its UIA tree only once asked
    auto = True  # considered when call.app = "auto"

    @property
    def keys(self) -> tuple[str, ...]:
        return (self.key,)

    def button(self, win: Any) -> Any | None:
        return uia.find_button(win, self.hangup, self.hangup_ids)

    def candidate(self, title: str) -> bool:
        return True

    def find(self, env: uia.Env) -> list[Call]:
        out = []
        for _, pid, win in env.of(self.exes):
            if not self.candidate(uia.text(win)):
                continue
            b = self.button(win)
            if b is not None:
                out.append(Call(self.key, win, pid, handle=b))
        return out

    def leave(self, env: uia.Env, call: Call, opts: Options) -> LeaveResult:
        keys = self.keys_for(opts)
        if keys:
            if self.focus_first:
                call.window.set_focus()
            env.send(keys)
            if uia.wait_for(lambda: self.button(call.window) is None, GONE_S):
                return LeaveResult(True, f"left the {self.name} call", app=self.name)
            log.info("%s: the shortcut didn't end the call; pressing its button", self.name)
        button = self.button(call.window) or call.handle
        uia.press(button)
        return self.after_press(env, call)

    focus_first = True

    def keys_for(self, opts: Options) -> str:
        return self.shortcut

    def after_press(self, env: uia.Env, call: Call) -> LeaveResult:
        return LeaveResult(True, f"left the {self.name} call", app=self.name)


class Teams(_ButtonApp):
    key, name = "teams", "Microsoft Teams"
    exes = ("ms-teams.exe", "teams.exe")
    hangup = re.compile(r"(?i)^(leave|hang up)$")
    hangup_ids = ("hangup-button", "splitButton-ddd__primaryActionButton")  # organizers get the split button
    shortcut = "^+h"  # "End audio/video call": leaves; ending the meeting for all is a separate menu
    on_hold = re.compile(r"(?i)^(resume|navigate back to call window)$")  # a call on hold / the mini window

    def button(self, win: Any) -> Any | None:
        # the AutomationIds don't depend on Teams' language; the name is a fallback for a redesigned toolbar
        found = None
        for b in uia.buttons(win):
            if self.on_hold.search(uia.text(b)):
                return None
            if found is None and (uia.auto_id(b) in self.hangup_ids or self.hangup.search(uia.text(b))):
                found = b
        return found


class Signal(_ButtonApp):
    key, name = "signal", "Signal"
    exes = ("signal.exe",)
    hangup = re.compile(r"(?i)^(end call|leave call|hang ?up)$")
    shortcut = "%+e"  # Signal's own "hang up" (Alt+Shift+E); E between keeps Windows' Alt+Shift layout switch off


class Webex(_ButtonApp):
    key, name = "webex", "Webex"
    exes = ("ciscocollabhost.exe", "webexhost.exe", "atmgr.exe")
    hangup = re.compile(r"(?i)^(leave( meeting)?|end( meeting)?)$")
    confirm = re.compile(r"(?i)^leave meeting$")

    def leave(self, env: uia.Env, call: Call, opts: Options) -> LeaveResult:
        call.window.set_focus()
        env.send("^l")  # "Leave or end the meeting": asks to confirm unless the user turned that off
        def confirm_or_gone():
            for _, pid, win in env.fresh(self.exes):
                b = uia.find_button(win, self.confirm)
                if b is not None:
                    return b
            return "gone" if self.button(call.window) is None else None

        got = uia.wait_for(confirm_or_gone, GONE_S)
        if got == "gone":
            return LeaveResult(True, "left the Webex meeting", app=self.name)
        if got is None:
            env.send("{ESC}")
            return LeaveResult(False, "Webex did not show 'Leave meeting'; leave from Webex yourself.", app=self.name)
        uia.press(got)
        return LeaveResult(True, "left the Webex meeting", app=self.name)


class Discord(_ButtonApp):
    key, name = "discord", "Discord"
    exes = ("discord.exe", "discordptb.exe", "discordcanary.exe")
    hangup = re.compile(r"(?i)^disconnect$")
    focus_first = False  # Discord's keybinds are global

    def keys_for(self, opts: Options) -> str:
        return opts.discord_keys


class Telegram(_ButtonApp):
    key, name = "telegram", "Telegram"
    exes = ("telegram.exe",)
    hangup = re.compile(r"(?i)^(end call|leave)$")  # 1:1 call / group video chat
    lazy = False  # Qt: a native UIA tree
    confirm = re.compile(r"(?i)^leave$")

    def candidate(self, title: str) -> bool:
        return not title.startswith("Telegram")  # calls get their own window, titled with the other person's name

    def after_press(self, env: uia.Env, call: Call) -> LeaveResult:
        if uia.text(call.handle).lower() != "leave":
            return LeaveResult(True, "ended the Telegram call", app=self.name)
        # a group video chat asks again; its "End video chat" box stays unticked, only "Leave" is pressed
        def confirm():
            for _, _, win in env.fresh(self.exes):
                if win is not call.window:
                    b = uia.find_button(win, self.confirm)
                    if b is not None:
                        return b
            return None

        b = uia.wait_for(confirm, 1.5)
        if b is not None:
            uia.press(b)
        return LeaveResult(True, "left the Telegram video chat", app=self.name)


class WhatsApp(_ButtonApp):
    key, name = "whatsapp", "WhatsApp"
    exes = ("whatsapp.root.exe", "whatsapp.exe")
    hangup = re.compile(r"(?i)^(end( video)? call|leave call|hang ?up)$")


class Slack(_ButtonApp):
    key, name = "slack", "Slack"
    exes = ("slack.exe",)
    hangup = re.compile(r"(?i)^leave( huddle)?$")
    auto = False  # only when call.app = "slack": a plain "Leave" button is too common to guess with

    def button(self, win: Any) -> Any | None:
        bs = uia.buttons(win)
        if not any("huddle" in uia.text(b).lower() for b in bs):
            return None
        return next((b for b in bs if self.hangup.search(uia.text(b))), None)


APPS = (Teams(), Telegram(), Signal(), Discord(), WhatsApp(), Webex(), Slack())
