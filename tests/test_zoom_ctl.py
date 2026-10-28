"""zoom_ctl against a fake pywinauto: no Zoom, no keystrokes reach the desktop."""

from __future__ import annotations

import logging
import sys
import types

import pytest

from zap.devices import zoom_ctl

ZOOM_PID, OTHER_PID = 4242, 777


class FakeButton:
    def __init__(self, title: str, ui: FakeUI) -> None:
        self.title, self.ui = title, ui

    def window_text(self) -> str:
        return self.title

    def invoke(self) -> None:
        self.ui.invoked.append(self.title)
        self.ui.on_invoke(self.title)


class FakeWindow:
    def __init__(self, cls: str, pid: int, ui: FakeUI, title: str = "", buttons=()) -> None:
        self.cls, self.pid, self.ui, self.title, self.buttons = cls, pid, ui, title, list(buttons)

    def process_id(self) -> int:
        return self.pid

    def class_name(self) -> str:
        return self.cls

    def window_text(self) -> str:
        return self.title

    def set_focus(self) -> None:
        self.ui.focused.append(self.cls)

    def descendants(self, control_type=None):
        assert control_type == "Button"
        return [FakeButton(t, self.ui) for t in self.buttons]


class FakeUI:
    """State of the pretend desktop; popups appear in response to keys and clicks."""

    def __init__(self, meeting_pid: int = ZOOM_PID, popup=(), after_invoke=()) -> None:
        self.keys: list[str] = []
        self.focused: list[str] = []
        self.invoked: list[str] = []
        self.meeting = FakeWindow("ConfMultiTabContentWndClass", meeting_pid, self, "Zoom Meeting", ["Mute"])
        self.top = [self.meeting, FakeWindow("Chrome_WidgetWin_1", OTHER_PID, self, "browser")]
        self.uia = [self.meeting]
        self.popup, self.after_invoke = list(popup), list(after_invoke)

    def send_keys(self, keys: str) -> None:
        self.keys.append(keys)
        if keys == "%q" and self.popup:
            self.uia = [self.meeting, FakeWindow("LeavePopup", ZOOM_PID, self, "", self.popup)]

    def on_invoke(self, title: str) -> None:
        if self.after_invoke:
            self.uia = [FakeWindow("AssignHostDlg", ZOOM_PID, self, "", self.after_invoke)]


def install(monkeypatch, ui: FakeUI) -> types.ModuleType:
    class Desktop:
        def __init__(self, backend: str) -> None:
            self.backend = backend

        def windows(self, class_name=None, process=None, visible_only=True, **kw):
            pool = ui.top if self.backend == "win32" else ui.uia
            return [w for w in pool if (class_name is None or w.cls == class_name)
                    and (process is None or w.pid == process)]

    mod = types.ModuleType("pywinauto")
    mod.Desktop = Desktop
    mod.application = types.SimpleNamespace(
        process_module=lambda pid: r"C:\Program Files\Zoom\bin\Zoom.exe" if pid == ZOOM_PID else r"C:\x\chrome.exe"
    )
    mod.keyboard = types.SimpleNamespace(send_keys=ui.send_keys)
    monkeypatch.setitem(sys.modules, "pywinauto", mod)
    monkeypatch.setattr(zoom_ctl, "POLL_S", 0.25)
    monkeypatch.setattr(zoom_ctl, "AFTER_S", 0.15)
    monkeypatch.setattr(zoom_ctl, "POLL_STEP_S", 0.01)
    return mod


def no_enter(ui: FakeUI) -> None:
    for k in ui.keys:
        assert "ENTER" not in k.upper() and "~" not in k


POPUP = ("End Meeting for All", "Leave Meeting", "Cancel")


def test_dry_run_never_touches_pywinauto(monkeypatch, caplog):
    touched = []

    class Tripwire(types.ModuleType):
        def __getattr__(self, name):
            touched.append(name)
            raise AssertionError("pywinauto touched during dry run")

    monkeypatch.setitem(sys.modules, "pywinauto", Tripwire("pywinauto"))
    with caplog.at_level(logging.WARNING, logger="zap.devices.zoom_ctl"):
        r1 = zoom_ctl.leave(dry_run=True)
        r2 = zoom_ctl.leave(end_for_all=True, dry_run=True)
    assert r1.ok and r1.dry_run and r2.ok and r2.dry_run
    assert touched == []
    assert "WOULD LEAVE" in caplog.text


def test_dry_run_works_without_pywinauto(monkeypatch):
    monkeypatch.setitem(sys.modules, "pywinauto", None)  # any import would raise ImportError
    assert zoom_ctl.leave().ok  # dry_run defaults to True


def test_leave_clicks_leave_meeting(monkeypatch):
    ui = FakeUI(popup=POPUP)
    install(monkeypatch, ui)
    r = zoom_ctl.leave(end_for_all=False, dry_run=False)
    assert r.ok and not r.dry_run
    assert ui.focused == ["ConfMultiTabContentWndClass"]
    assert ui.keys == ["%q"]
    assert ui.invoked == ["Leave Meeting"]
    no_enter(ui)


def test_end_for_all_clicks_end_meeting(monkeypatch):
    ui = FakeUI(popup=POPUP)
    install(monkeypatch, ui)
    r = zoom_ctl.leave(end_for_all=True, dry_run=False)
    assert r.ok
    assert ui.invoked == ["End Meeting for All"]
    no_enter(ui)


def test_regex_per_mode():
    assert zoom_ctl.LEAVE_RE.search("Leave Meeting")
    assert zoom_ctl.LEAVE_RE.search("LEAVE MEETING")
    assert not zoom_ctl.LEAVE_RE.search("Assign a new host and leave meeting")
    assert not zoom_ctl.LEAVE_RE.search("End Meeting for All")
    assert zoom_ctl.END_ALL_RE.search("End Meeting for All")
    assert not zoom_ctl.END_ALL_RE.search("Leave Meeting")


def test_no_button_sends_escape(monkeypatch):
    ui = FakeUI(popup=())
    install(monkeypatch, ui)
    r = zoom_ctl.leave(dry_run=False)
    assert not r.ok and "Leave meeting" in r.message
    assert ui.keys == ["%q", "{ESC}"]
    assert ui.invoked == []
    no_enter(ui)


def test_end_for_all_missing_sends_escape(monkeypatch):
    ui = FakeUI(popup=("Leave Meeting",))  # not the host: no "End meeting for all"
    install(monkeypatch, ui)
    r = zoom_ctl.leave(end_for_all=True, dry_run=False)
    assert not r.ok
    assert ui.invoked == [] and ui.keys == ["%q", "{ESC}"]


def test_assign_host_dialog_sends_escape(monkeypatch):
    ui = FakeUI(popup=POPUP, after_invoke=("Assign and Leave", "Cancel"))
    install(monkeypatch, ui)
    r = zoom_ctl.leave(dry_run=False)
    assert not r.ok and "host" in r.message
    assert ui.invoked == ["Leave Meeting"]
    assert ui.keys == ["%q", "{ESC}"]
    no_enter(ui)


def test_no_meeting_window(monkeypatch):
    ui = FakeUI(meeting_pid=OTHER_PID, popup=POPUP)  # meeting class but not Zoom.exe
    install(monkeypatch, ui)
    r = zoom_ctl.leave(dry_run=False)
    assert not r.ok and "No Zoom meeting window" in r.message
    assert ui.keys == [] and ui.invoked == []


def test_leave_never_raises(monkeypatch):
    ui = FakeUI(popup=POPUP)
    mod = install(monkeypatch, ui)

    class Broken:
        def __init__(self, backend):
            raise RuntimeError("COM exploded")

    mod.Desktop = Broken
    r = zoom_ctl.leave(dry_run=False)
    assert not r.ok and "COM exploded" in r.message


def test_send_refuses_enter():
    sent = []
    for keys in ("{ENTER}", "~", "%{enter}"):
        with pytest.raises(ValueError):
            zoom_ctl._send(sent.append, keys)
    assert sent == []


def test_probe_lists_zoom_windows_only(monkeypatch):
    ui = FakeUI(popup=POPUP)
    install(monkeypatch, ui)
    rows = zoom_ctl.probe()
    assert len(rows) == 1
    assert rows[0]["class"] == "ConfMultiTabContentWndClass" and rows[0]["meeting"]
    assert rows[0]["buttons"] == ["Mute"]
    assert "meeting window" in zoom_ctl.format_probe(rows)
    assert "no visible Zoom" in zoom_ctl.format_probe([])
