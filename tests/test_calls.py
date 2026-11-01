"""Leaving calls in every app (zap.calls) against a fake desktop: no app runs, no keystroke reaches Windows."""

from __future__ import annotations

import sys
import types

import pytest

from zap import calls, config
from zap.calls import uia


class El:
    """A pretend UI Automation element (window, button, tab, checkbox)."""

    def __init__(self, ui: Desk, kind: str = "Button", name: str = "", auto: str = "", kids=(), pid: int = 0,
                 on_invoke=None, cls: str = "") -> None:
        self.ui, self.kind, self.name, self.auto, self.kids, self.pid = ui, kind, name, auto, list(kids), pid
        self.on_invoke, self.cls, self.selected = on_invoke, cls, False
        self.element_info = types.SimpleNamespace(control_type=kind)
        self.handle = id(self)

    def window_text(self) -> str:
        return self.name

    def automation_id(self) -> str:
        return self.auto

    def process_id(self) -> int:
        return self.pid

    def class_name(self) -> str:
        return self.cls

    def children(self, **kw):
        return list(self.kids)

    def descendants(self, control_type=None, **kw):
        out, todo = [], list(self.kids)
        while todo:
            k = todo.pop(0)
            out.append(k)
            todo.extend(k.kids)
        return [k for k in out if control_type is None or k.kind == control_type]

    def invoke(self) -> None:
        self.ui.log.append(("press", self.name or self.auto))
        if self.on_invoke:
            self.on_invoke()

    def set_focus(self) -> None:
        self.ui.log.append(("focus", self.name))

    def select(self) -> None:
        if self.ui.selectable:
            self.selected = True
        self.ui.log.append(("select", self.name))

    def is_selected(self) -> bool:
        return self.selected

    def close(self) -> None:
        self.ui.log.append(("close", self.name))


class Desk:
    """The pretend desktop: top-level windows, the program of each pid, and what keys do."""

    def __init__(self) -> None:
        self.windows: list[El] = []
        self.exes: dict[int, str] = {}
        self.log: list[tuple[str, str]] = []
        self.on_keys: dict[str, object] = {}
        self.selectable = True

    def window(self, exe: str, title: str, kids=(), cls: str = "") -> El:
        pid = 100 + len(self.windows)
        self.exes[pid] = rf"C:\Apps\{exe}"
        w = El(self, "Window", title, kids=kids, pid=pid, cls=cls)
        for k in w.descendants():
            k.pid = pid
        self.windows.append(w)
        return w

    def button(self, name: str = "", auto: str = "", on_invoke=None) -> El:
        return El(self, "Button", name, auto, on_invoke=on_invoke)

    def send_keys(self, keys: str) -> None:
        self.log.append(("keys", keys))
        if keys in self.on_keys:
            self.on_keys[keys]()

    def pressed(self) -> list[str]:
        return [n for kind, n in self.log if kind == "press"]

    def keys(self) -> list[str]:
        return [n for kind, n in self.log if kind == "keys"]


@pytest.fixture
def desk(monkeypatch) -> Desk:
    d = Desk()

    class Desktop:
        def __init__(self, backend: str) -> None:
            self.backend = backend

        def windows(self, class_name=None, process=None, visible_only=True, **kw):
            return [w for w in d.windows if (class_name is None or w.cls == class_name)
                    and (process is None or w.pid == process)]

    mod = types.ModuleType("pywinauto")
    mod.Desktop = Desktop
    mod.application = types.SimpleNamespace(process_module=lambda pid: d.exes.get(pid, ""))
    mod.keyboard = types.SimpleNamespace(send_keys=d.send_keys)
    monkeypatch.setitem(sys.modules, "pywinauto", mod)
    monkeypatch.setattr(calls, "PLATFORM", "win32")
    monkeypatch.setattr(uia, "POLL_STEP_S", 0.005)
    monkeypatch.setattr(uia, "WEB_WAIT_S", 0.05)
    from zap.calls import apps

    monkeypatch.setattr(apps, "GONE_S", 0.05)
    return d


def leave(app: str = "auto", **kw) -> calls.LeaveResult:
    return calls.leave(app=app, dry_run=False, **kw)


def teams_window(d: Desk, ends_with_shortcut: bool = True, extra=()) -> El:
    hang = d.button("Leave", "hangup-button")
    win = d.window("ms-teams.exe", "Meeting | Microsoft Teams",
                   [d.button("Mic", "microphone-button"), hang, *extra])
    if ends_with_shortcut:
        d.on_keys["^+h"] = lambda: win.kids.remove(hang)
    hang.on_invoke = lambda: win.kids.remove(hang) if hang in win.kids else None
    return win


# ---------------------------------------------------------------- desktop apps
def test_teams_leaves_with_its_shortcut(desk):
    teams_window(desk)
    r = leave()
    assert r.ok and r.app == "Microsoft Teams"
    assert desk.keys() == ["^+h"] and desk.pressed() == []
    assert ("focus", "Meeting | Microsoft Teams") in desk.log


def test_teams_presses_leave_when_the_shortcut_does_nothing(desk):
    teams_window(desk, ends_with_shortcut=False)
    assert leave().ok
    assert desk.keys() == ["^+h"] and desk.pressed() == ["Leave"]


def test_teams_call_on_hold_is_not_a_call(desk):
    teams_window(desk, extra=[desk.button("Resume")])
    r = leave()
    assert not r.ok and "No call found" in r.message and desk.keys() == []


def test_signal_hangs_up_with_alt_shift_e(desk):
    hang = desk.button("End call")
    win = desk.window("Signal.exe", "Signal", [hang])
    desk.on_keys["%+e"] = lambda: win.kids.remove(hang)
    assert leave().ok and desk.keys() == ["%+e"]


def test_discord_uses_the_users_keybind_or_the_button(desk):
    hang = desk.button("Disconnect")
    win = desk.window("Discord.exe", "Discord", [hang])
    desk.on_keys["^+{F9}"] = lambda: win.kids.remove(hang)
    assert leave(discord_keys="ctrl+shift+f9").ok
    assert desk.keys() == ["^+{F9}"] and desk.pressed() == []
    assert not any(kind == "focus" for kind, _ in desk.log)  # Discord's keybinds are global

    desk.log.clear()
    win.kids.append(hang)
    assert leave().ok and desk.pressed() == ["Disconnect"] and desk.keys() == []


def test_telegram_ends_a_call_but_its_main_window_is_not_one(desk):
    desk.window("Telegram.exe", "Telegram", [desk.button("Leave")])  # e.g. a channel's "Leave": not a call
    assert not leave().ok
    desk.window("Telegram.exe", "Alice", [desk.button("Mute"), desk.button("End Call")])
    r = leave()
    assert r.ok and desk.pressed() == ["End Call"]


def test_telegram_group_chat_confirms_leave_and_never_ends_it(desk):
    box = El(desk, "CheckBox", "End video chat")

    def ask():
        desk.window("Telegram.exe", "Leave video chat", [box, desk.button("Cancel"), desk.button("Leave")])

    desk.window("Telegram.exe", "Study group", [desk.button("Leave", on_invoke=ask)])
    assert leave().ok
    assert desk.pressed() == ["Leave", "Leave"]  # the chat's button, then the confirmation; the box untouched


def test_whatsapp_presses_end_call(desk):
    desk.window("WhatsApp.Root.exe", "WhatsApp", [desk.button("End call")])
    r = leave()
    assert r.ok and r.app == "WhatsApp" and desk.pressed() == ["End call"]


def test_webex_confirms_leave_meeting_never_end_for_all(desk):
    def confirm():
        desk.window("CiscoCollabHost.exe", "", [desk.button("End meeting for all"), desk.button("Leave meeting"),
                                                desk.button("Cancel")])

    desk.window("CiscoCollabHost.exe", "Webex meeting", [desk.button("Leave")])
    desk.on_keys["^l"] = confirm
    assert leave().ok
    assert desk.keys() == ["^l"] and desk.pressed() == ["Leave meeting"]


def test_slack_only_when_chosen(desk):
    desk.window("slack.exe", "Slack", [desk.button("Huddle with Sam"), desk.button("Leave")])
    assert not leave().ok and desk.pressed() == []  # "Leave" alone is too common to guess with
    assert leave("slack").ok and desk.pressed() == ["Leave"]


# ---------------------------------------------------------------- browsers
def tab(d: Desk, name: str, buttons=("Close",), on_close=None) -> El:
    return El(d, "TabItem", name, kids=[d.button(b, on_invoke=on_close if b == "Close" else None) for b in buttons])


def test_browser_closes_just_the_call_tab(desk):
    meet = tab(desk, "Meet - abc-defg-hij - Camera and microphone recording", ("Mute tab", "Close"))
    strip = El(desk, "Tab", "Tab strip", kids=[tab(desk, "Inbox - Gmail"), meet])
    desk.window("chrome.exe", "Meet - abc-defg-hij - Google Chrome", [strip])
    r = leave()
    assert r.ok and "Google Meet in Chrome" in r.message
    assert desk.pressed() == ["Close"] and desk.keys() == []


def test_browser_ignores_buttons_inside_the_page(desk):
    page = El(desk, "Document", "Teams", kids=[El(desk, "TabItem", "Chat - Camera recording")])  # a page's own tab
    desk.window("msedge.exe", "Teams - Edge", [El(desk, "Tab", "", kids=[tab(desk, "News")]), page])
    assert not leave().ok


def test_teams_in_a_browser_presses_its_leave_button(desk):
    t = tab(desk, "Meeting | Microsoft Teams - Camera and microphone recording")
    page = El(desk, "Document", "Teams", kids=[desk.button("Leave", "hangup-button")])
    desk.window("msedge.exe", "Microsoft Teams - Edge", [El(desk, "Tab", "", kids=[t]), page])
    r = leave("teams")
    assert r.ok and desk.pressed() == ["Leave"] and ("select", t.name) in desk.log


def test_pinned_call_tab_is_closed_only_when_surely_selected(desk):
    t = tab(desk, "Jitsi Meet - Microphone recording", buttons=())  # pinned: no close button
    desk.window("chrome.exe", "Jitsi Meet", [El(desk, "Tab", "", kids=[t])])
    assert leave().ok and desk.keys() == ["^w"]

    desk.log.clear()
    t.selected, desk.selectable = False, False  # the tab can't be selected: Ctrl+W would close another tab
    assert not leave().ok and desk.keys() == []


def test_firefox_meet_tab_counts_without_the_recording_mark(desk):
    desk.window("firefox.exe", "Meet - Mozilla Firefox", [El(desk, "Tab", "", kids=[tab(desk, "Meet - abc-defg")])])
    assert leave("meet").ok and desk.pressed() == ["Close"]


def test_installed_meet_app_window_is_closed(desk):
    desk.window("chrome.exe", "Meet - abc-defg-hij")  # a PWA window: no tab strip
    assert leave().ok and ("close", "Meet - abc-defg-hij") in desk.log


def test_a_whatsapp_tab_without_a_call_is_not_a_call(desk):
    desk.window("chrome.exe", "WhatsApp", [El(desk, "Tab", "", kids=[tab(desk, "(3) WhatsApp")])])
    assert not leave().ok and desk.pressed() == []


# ---------------------------------------------------------------- choosing the call
def test_two_calls_leave_neither(desk):
    teams_window(desk)
    strip = El(desk, "Tab", "", kids=[tab(desk, "Meet - x - Camera recording")])
    desk.window("chrome.exe", "Meet", [strip])
    r = leave()
    assert not r.ok and "more than one call" in r.message and "Microsoft Teams" in r.message
    assert desk.keys() == [] and desk.pressed() == []
    assert leave("teams").ok and desk.keys() == ["^+h"]  # naming the app settles it


def test_no_call_app_running(desk):
    r = leave()
    assert not r.ok and "No call found" in r.message


def test_dry_run_and_other_platforms_touch_nothing(desk, monkeypatch):
    teams_window(desk)
    assert calls.leave(dry_run=True).dry_run and desk.log == []
    monkeypatch.setattr(calls, "PLATFORM", "darwin")
    r = leave()
    assert not r.ok and "Windows only" in r.message and desk.log == []


def test_a_broken_desktop_never_raises(desk, monkeypatch):
    def boom(backend):
        raise RuntimeError("COM exploded")

    monkeypatch.setattr(sys.modules["pywinauto"], "Desktop", boom)
    r = leave()
    assert not r.ok and "COM exploded" in r.message


def test_end_for_everyone_is_never_pressed():
    d = Desk()
    for name in ("End meeting for all", "End meeting", "Für alle beenden", "End video chat", "结束会议"):
        with pytest.raises(ValueError):
            uia.press(El(d, name=name))
    assert d.pressed() == []
    uia.press(El(d, name="End meeting for all"), allow_end_for_all=True)  # Zoom's end_for_all setting only
    assert d.pressed() == ["End meeting for all"]


def test_keys_as_people_write_them():
    assert calls.keys_from_combo("ctrl+shift+f9") == "^+{F9}"
    assert calls.keys_from_combo("Alt + D") == "%d"
    assert calls.keys_from_combo("") == ""
    for bad in ("ctrl+enter", "hyper+x", "ctrl+"):
        with pytest.raises(ValueError):
            calls.keys_from_combo(bad)


# ---------------------------------------------------------------- config
def test_old_zoom_section_still_works(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text("[zoom]\ndry_run = false\ncountdown_s = 5.0\n", encoding="utf-8")
    cfg = config.load(f)
    assert cfg.call.dry_run is False and cfg.call.countdown_s == 5.0 and cfg.call.app == "auto"
    f.write_text("[zoom]\ndry_run = false\n[call]\napp = \"teams\"\n", encoding="utf-8")
    with pytest.raises(config.ConfigError, match="old name"):
        config.load(f)


def test_call_settings_are_checked(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text("[call]\napp = \"temas\"\n", encoding="utf-8")
    with pytest.raises(config.ConfigError, match="teams"):
        config.load(f)
    f.write_text("[call]\ndiscord_keys = \"ctrl+enter\"\n", encoding="utf-8")
    with pytest.raises(config.ConfigError, match="discord_keys"):
        config.load(f)
