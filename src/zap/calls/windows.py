"""Windows: find the call (in whichever app is running one) and leave it, by UI Automation."""

from __future__ import annotations

import logging
import time

from zap.calls import apps, browser, uia, zoom
from zap.calls.base import APPS, Call, LeaveResult, Options, name_of

log = logging.getLogger(__name__)

STRATEGIES = (zoom.Zoom(), *apps.APPS, browser.Browsers())
MAX_PROBE_BUTTONS = 60


def wanted(call: Call, app: str) -> bool:
    if app == "auto":
        return True
    if app == "browser":
        return call.where != "app"
    return call.app == app


def find_calls(env: uia.Env, app: str = "auto") -> list[tuple[object, Call]]:
    """(app strategy, call) for every call found for ``app`` ("auto": in every app). Apps with web
    content are asked again for up to uia.WEB_WAIT_S: their UIA tree appears only after a first look."""
    pending = [s for s in STRATEGIES if env.of(s.exes) and (s.auto if app == "auto" else app in s.keys)]
    found: list[tuple[object, Call]] = []
    deadline = time.perf_counter() + uia.WEB_WAIT_S
    while pending:
        for s in list(pending):
            calls = [c for c in s.find(env) if wanted(c, app)]
            if calls or not s.lazy:
                found += [(s, c) for c in calls]
                pending.remove(s)
        if not pending or time.perf_counter() >= deadline:
            break
        time.sleep(uia.POLL_STEP_S)
    return found


def leave(app: str, opts: Options) -> LeaveResult:
    env = uia.look()
    found = find_calls(env, app)
    if not found:
        where = "any call app" if app == "auto" else name_of(app)
        return LeaveResult(False, f"No call found in {where}; hang up yourself.")
    if len(found) > 1:
        names = ", ".join(sorted({_label(c) for _, c in found}))
        return LeaveResult(False, f"Found more than one call ({names}); not leaving any. Set call.app to the app "
                                  "you meet in, or hang up yourself.")
    strategy, call = found[0]
    log.info("leaving %s", _label(call))
    return strategy.leave(env, call, opts)


def _label(call: Call) -> str:
    name = APPS.get(call.app, call.app)
    return name if call.where == "app" else f"{name} in {call.where}"


def probe() -> list[dict]:
    """The visible windows of every call app ZAP knows, with their buttons and calls (``zap doctor --calls``)."""
    env = uia.look()
    exes = {e for s in STRATEGIES for e in s.exes}
    calls = [c for _, c in find_calls(env)]
    rows = []
    for exe, pid, win in env.windows:
        if exe not in exes:
            continue
        names = []
        for b in uia.buttons(win)[:MAX_PROBE_BUTTONS]:
            t, i = uia.text(b), uia.auto_id(b)
            if t or i:
                names.append(f"{t}#{i}" if i else t)
        tabs = [uia.text(t) for t in uia.walk(win, ("TabItem",))] if exe in browser.BROWSERS else []
        rows.append({"exe": exe, "pid": pid, "title": uia.text(win), "buttons": names, "tabs": tabs,
                     "calls": [_label(c) for c in calls if _same(c.window, win)]})
    return rows


def _same(a, b) -> bool:
    """The same window, even through two pywinauto backends (Zoom's is found as a win32 window)."""
    ha, hb = getattr(a, "handle", None), getattr(b, "handle", None)
    return a is b or (ha is not None and ha == hb)


def format_probe(rows: list[dict]) -> str:
    if not rows:
        return "  no call app windows open (start your call app and join a call, then run this again)"
    lines = []
    for r in rows:
        tag = f"  [call: {', '.join(r['calls'])}]" if r["calls"] else ""
        lines.append(f"  {r['exe']} (pid {r['pid']})  title={r['title']!r}{tag}")
        if r["tabs"]:
            lines.append("    tabs: " + ", ".join(repr(t) for t in r["tabs"]))
        if r["buttons"]:
            lines.append("    buttons: " + ", ".join(repr(b) for b in r["buttons"]))
    return "\n".join(lines)
