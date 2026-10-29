from __future__ import annotations

import pytest

from zap.clock import ReplayClock
from zap.config import defaults
from zap.gestures.arbiter import PANIC, Arbiter
from zap.gestures.dwell import COOLDOWN, IDLE, WAIT_RELEASE
from zap.gestures.types import (
    CANCEL,
    COVER_FACE,
    EYEBROWS_UP,
    FOREHEAD,
    HANDS_UP,
    SNAP1,
    SNAP2,
    SNAP3,
    THUMBS_UP,
    Intent,
    SnapBurst,
)

DT = 1 / 30


def make(armed=True, grace=0.0, leave_guard=0.0, confirm="countdown", gestures=None) -> Arbiter:
    cfg = defaults()
    cfg.app.armed_on_start = armed
    cfg.app.arm_grace_s = grace
    cfg.zoom.leave_guard_s = leave_guard
    cfg.zoom.confirm = confirm
    cfg.gestures = gestures or {"eyebrows_up": {"t_on": 1.0}}  # timings below assume a 1 s brow hold
    return Arbiter(cfg, ReplayClock(0.0))


def run(arb: Arbiter, truths, t0: float, t1: float, events=None, **kw) -> list[Intent]:
    """Frames at k*DT for t in [t0, t1). ``truths``/``events``/kwargs may be callables of t."""
    out = []
    for k in range(round(t0 / DT), round(t1 / DT)):
        t = k * DT
        tr = truths(t) if callable(truths) else truths
        ev = events(t) if callable(events) else (events or [])
        args = {key: (v(t) if callable(v) else v) for key, v in kw.items()}
        out += arb.update(tr, ev, t, **args)
    return out


def at(t0: float):
    """Events only on the frame nearest t0."""
    k0 = round(t0 / DT)
    return lambda evs: (lambda t: evs if round(t / DT) == k0 else [])


def kinds(intents) -> list[tuple[str, str]]:
    return [(i.kind, i.gesture) for i in intents]


# ---------------------------------------------------------------- basics
def test_hands_up_trigger_after_dwell_and_settle_then_release():
    arb = make()
    out = run(arb, lambda t: {HANDS_UP: t < 1.0}, 0.0, 4.0)
    assert kinds(out) == [("trigger", HANDS_UP), ("release", HANDS_UP)]
    assert out[0].t == pytest.approx(0.4 + 0.1, abs=DT + 1e-6)
    assert out[1].t == pytest.approx(0.4 + 2.5, abs=DT + 1e-6)  # min_show beats the drop at 1.0


def test_disarmed_ignores_gestures_and_snaps():
    arb = make(armed=False)
    out = run(arb, {HANDS_UP: True, EYEBROWS_UP: True}, 0.0, 3.0, events=at(1.0)([SnapBurst(1, 1.0, 1.0)]))
    assert out == []
    assert arb.status(3.0)["armed"] is False
    assert arb.status(3.0)["suppressed"] == {HANDS_UP: "disarmed", EYEBROWS_UP: "disarmed"}


def test_grace_after_arming():
    arb = make(armed=False, grace=3.0)
    assert arb.command("arm_toggle", 0.0) == []
    assert run(arb, {FOREHEAD: True}, 0.0, 2.9, events=at(1.0)([SnapBurst(1, 1.0, 1.0)])) == []
    out = run(arb, {FOREHEAD: True}, 2.9, 5.0)
    assert kinds(out) == [("trigger", FOREHEAD)]
    assert out[0].t == pytest.approx(3.0 + 0.7 + 0.1, abs=2 * DT)


def test_manual_triggers_work_disarmed():
    arb = make(armed=False)
    assert kinds(arb.command("trigger:snap3", 1.0)) == [("trigger", SNAP3)]
    assert kinds(arb.command("trigger:snap2", 1.0)) == [("trigger", SNAP2)]  # deliberate: no leave guard
    assert arb.command("trigger:nonsense", 1.0) == []
    assert arb.command("sound_toggle", 1.0) == []  # not ours


def test_manual_hands_up_toggles():
    arb = make(armed=False)
    assert kinds(arb.command("trigger:hands_up", 1.0)) == [("trigger", HANDS_UP)]
    assert arb.status(1.0)["holding"] == [HANDS_UP]
    assert kinds(arb.command("trigger:hands_up", 2.0)) == [("release", HANDS_UP)]
    assert arb.status(2.0)["holding"] == []


# ---------------------------------------------------------------- snaps
@pytest.mark.parametrize(("count", "gesture"), [(1, SNAP1), (2, SNAP2), (3, SNAP3)])
def test_snap_counts(count, gesture):
    arb = make()
    out = arb.update({}, [SnapBurst(count, 1.0, 1.2, (100.0, 200.0))], 1.3)
    assert out == [Intent("trigger", gesture, 1.3, (100.0, 200.0))]


def test_odd_burst_ignored():
    arb = make()
    assert arb.update({}, [SnapBurst(5, 1.0, 1.2)], 1.3) == []


def test_leave_guard_blocks_snap2_only():
    arb = make(leave_guard=60.0)
    assert arb.update({}, [SnapBurst(2, 10.0, 10.2)], 10.3) == []
    assert arb.status(10.3)["suppressed"] == {SNAP2: "leave guard"}
    assert kinds(arb.update({}, [SnapBurst(3, 11.0, 11.2)], 11.3)) == [("trigger", SNAP3)]
    assert kinds(arb.update({}, [SnapBurst(2, 61.0, 61.2)], 61.3)) == [("trigger", SNAP2)]


def test_leave_guard_counts_from_rearming():
    arb = make(leave_guard=60.0)
    arb.command("disarm", 100.0)
    arb.command("arm", 101.0)
    assert arb.update({}, [SnapBurst(2, 120.0, 120.2)], 120.3) == []
    assert arb.status(120.3)["leave_guard_left"] == pytest.approx(40.7, abs=0.1)


@pytest.mark.parametrize("marker", ["charge", ("charge", (5.0, 6.0)), Intent("charge", where=(5.0, 6.0))])
def test_charge_discharge_markers(marker):
    arb = make()
    out = arb.update({}, [marker], 1.0)
    assert kinds(out) == [("charge", "")]
    assert kinds(arb.update({}, ["discharge"], 1.5)) == [("discharge", "")]
    assert arb.update({}, ["discharge"], 1.6) == []  # nothing left to fade


def test_charge_then_guarded_snap2_discharges():
    arb = make(leave_guard=60.0)
    arb.update({}, [("charge", (1.0, 2.0))], 1.0)
    assert kinds(arb.update({}, [SnapBurst(2, 1.0, 1.3)], 1.9)) == [("discharge", "")]


def test_charge_needs_armed_and_visible_or_vanished():
    arb = make(armed=False)
    assert arb.update({}, ["charge"], 1.0) == []
    arb = make()
    arb.presence_state = "dissolving"
    assert arb.update({}, ["charge"], 1.0) == []
    arb.presence_state = "vanished"
    assert kinds(arb.update({}, ["charge"], 2.0)) == [("charge", "")]


def test_snaps_pass_while_vanished_but_poses_do_not():
    arb = make()
    arb.presence_state = "vanished"
    out = run(arb, {HANDS_UP: True}, 0.0, 2.0, events=at(1.0)([SnapBurst(1, 1.0, 1.0)]))
    assert kinds(out) == [("trigger", SNAP1)]
    assert arb.status(2.0)["suppressed"][HANDS_UP] == "vanished"


# ---------------------------------------------------------------- modal
def test_modal_turns_snaps_into_cancel_and_blocks_poses():
    arb = make()
    arb.modal_active = True
    out = run(arb, {HANDS_UP: True, EYEBROWS_UP: True}, 0.0, 3.0, events=at(2.0)([SnapBurst(1, 2.0, 2.0)]))
    assert kinds(out) == [("cancel", SNAP1)]


def test_modal_ignores_cancels_while_the_snapping_hand_settles():
    arb = make()
    arb.modal_active = True
    out = run(arb, {CANCEL: True}, 0.0, 1.4, events=at(1.0)([SnapBurst(1, 1.0, 1.0)]))
    assert out == []  # the hand that just double-snapped is still open: not a cancel


def test_modal_open_palm_does_not_cancel_by_default():
    arb = make()
    arb.modal_active = True
    assert run(arb, {CANCEL: True}, 0.0, 4.0) == []


def test_modal_open_palm_cancels():
    arb = make()
    arb.cfg.zoom.palm_cancels = True
    run(arb, {}, 0.0, 0.5)
    arb.modal_active = True
    out = run(arb, lambda t: {CANCEL: t >= 2.1}, 0.5, 3.5)
    assert kinds(out) == [("cancel", CANCEL)]
    assert out[0].t == pytest.approx(2.1 + 0.8, abs=DT + 1e-6)


def test_palm_open_when_modal_starts_must_be_lowered_first():
    arb = make()
    arb.cfg.zoom.palm_cancels = True
    run(arb, {CANCEL: True}, 0.0, 1.0)  # hand already open after snapping
    arb.modal_active = True
    assert run(arb, {CANCEL: True}, 1.0, 2.0) == []
    out = run(arb, lambda t: {CANCEL: t >= 2.5}, 2.0, 3.5)  # lowered for 0.5 s, raised again
    assert kinds(out) == [("cancel", CANCEL)]


def test_open_palm_without_modal_does_nothing():
    arb = make()
    assert run(arb, {CANCEL: True}, 0.0, 2.0) == []
    assert CANCEL not in arb.status(2.0)["suppressed"]


def test_modal_cancel_works_even_disarmed():
    arb = make(armed=False)
    arb.modal_active = True
    arb.update({}, [], 0.0)
    assert kinds(arb.update({}, [SnapBurst(1, 2.0, 2.0)], 2.1)) == [("cancel", SNAP1)]


def test_thumbs_up_confirm_only_in_thumbs_up_mode():
    arb = make(confirm="countdown")
    arb.modal_active = True
    assert run(arb, {THUMBS_UP: True}, 0.0, 2.0) == []
    arb = make(confirm="thumbs_up")
    arb.modal_active = True
    out = run(arb, lambda t: {THUMBS_UP: t >= 0.5}, 0.0, 3.0)  # raised after the modal started
    assert kinds(out) == [("confirm", THUMBS_UP)]
    assert out[0].t == pytest.approx(1.5 + 0.6, abs=DT + 1e-6)  # counted once the 1.5 s grace is over


def test_thumbs_up_already_shown_when_modal_starts_needs_a_fresh_one():
    arb = make(confirm="thumbs_up")
    arb.modal_active = True
    assert run(arb, {THUMBS_UP: True}, 0.0, 2.0) == []


def test_confirm_command():
    arb = make(armed=False)
    assert arb.command("confirm", 1.0) == []
    arb.modal_active = True
    assert kinds(arb.command("confirm", 1.0)) == [("confirm", THUMBS_UP)]


# ---------------------------------------------------------------- panic / arming
def test_panic_drops_everything_and_pauses_gestures():
    arb = make()
    run(arb, {HANDS_UP: True}, 0.0, 1.0)
    arb.modal_active = True
    out = arb.command(PANIC, 1.0)
    assert out == [Intent(PANIC, "", 1.0)]
    assert arb.armed is True and arb.arm_time == 1.0  # still armed; the arm grace restarts
    assert all(d.state == IDLE for d in arb.dwells.values())
    arb.modal_active = False
    assert run(arb, {HANDS_UP: False}, 1.0, 5.0) == []  # no stray release after panic


def test_disarm_releases_held_effect():
    arb = make()
    run(arb, {HANDS_UP: True}, 0.0, 1.0)
    assert kinds(arb.command("arm_toggle", 1.0)) == [("release", HANDS_UP)]
    assert arb.armed is False


# ---------------------------------------------------------------- priorities & suppression
def test_hands_up_arming_suppresses_forehead():
    arb = make()
    out = run(arb, lambda t: {FOREHEAD: True, HANDS_UP: t >= 0.2}, 0.0, 2.0)
    assert kinds(out) == [("trigger", HANDS_UP)]
    assert arb.status(1.0)["suppressed"].get(FOREHEAD) == "hands up"


def test_lower_priority_waits_for_higher_arming_then_loses():
    arb = make()
    out = run(arb, lambda t: {EYEBROWS_UP: True, FOREHEAD: t >= 0.95}, 0.0, 3.0)
    assert kinds(out) == [("trigger", FOREHEAD)]
    assert out[0].t == pytest.approx(0.95 + 0.7 + 0.1, abs=2 * DT)
    assert arb.dwells[EYEBROWS_UP].state == WAIT_RELEASE  # superseded: must be let go


def test_face_only_pose_fires_after_settle():
    arb = make()
    out = run(arb, {EYEBROWS_UP: True}, 0.0, 2.0)
    assert kinds(out) == [("trigger", EYEBROWS_UP)]
    assert out[0].t == pytest.approx(1.0 + 0.1, abs=DT + 1e-6)


def test_t_on_override_from_detector():
    arb = make()
    out = run(arb, {EYEBROWS_UP: True}, 0.0, 3.0, t_on={EYEBROWS_UP: 2.0})
    assert out[0].t == pytest.approx(2.0 + 0.1, abs=DT + 1e-6)


def test_settle_rejects_a_pose_abandoned_right_after_on():
    arb = make(gestures={"forehead": {"t_off": 0.05}})
    # held exactly until "on" (0.7) then dropped: never survives the settle
    out = run(arb, lambda t: {FOREHEAD: t < 0.72}, 0.0, 2.0)
    assert out == []
    assert arb.dwells[FOREHEAD].state == IDLE  # nothing shown, so no cooldown


def test_snap_loaded_suppresses_hand_poses_for_600ms():
    arb = make()
    out = run(arb, {FOREHEAD: True}, 0.0, 3.0, snap_loaded=lambda t: abs(t - 0.5) < 1e-6 or round(t / DT) == 15)
    assert kinds(out) == [("trigger", FOREHEAD)]
    assert out[0].t == pytest.approx(0.5 + 0.6 + 0.7 + 0.1, abs=2 * DT)


def test_snap_loaded_list_form_and_eyebrows_unaffected():
    arb = make()
    out = run(arb, {EYEBROWS_UP: True}, 0.0, 2.0, snap_loaded=[(1.0, 2.0)])
    assert kinds(out) == [("trigger", EYEBROWS_UP)]
    arb = make()
    assert run(arb, {FOREHEAD: True}, 0.0, 2.0, snap_loaded=[(1.0, 2.0)]) == []


def test_pending_charge_blocks_pose_until_burst():
    arb = make()
    out = run(arb, {EYEBROWS_UP: True}, 0.0, 3.0, events=at(0.9)(["charge"]))
    # charge at 0.9 holds the eyebrows candidate (on at 1.0) until the charge times out at 2.4
    assert kinds(out) == [("charge", ""), ("trigger", EYEBROWS_UP)]
    assert 2.4 < out[1].t < 2.5
    arb = make()
    ev = {round(0.9 / DT): ["charge"], round(1.2 / DT): [SnapBurst(1, 0.9, 0.9)]}
    out = run(arb, {EYEBROWS_UP: True}, 0.0, 3.0, events=lambda t: ev.get(round(t / DT), []))
    assert kinds(out) == [("charge", ""), ("trigger", SNAP1), ("trigger", EYEBROWS_UP)]
    assert out[2].t > out[1].t


def test_snap_wins_the_frame():
    arb = make()
    k_fire = round(1.1 / DT)
    out = run(arb, {EYEBROWS_UP: True}, 0.0, 2.0, events=lambda t: [SnapBurst(3, t, t)] if round(t / DT) == k_fire else [])
    assert kinds(out) == [("trigger", SNAP3), ("trigger", EYEBROWS_UP)]
    assert out[1].t > out[0].t


def test_busy_holds_then_gap_then_fires():
    arb = make()
    out = run(arb, {FOREHEAD: True}, 0.0, 3.0, busy=lambda t: t < 1.0)
    assert kinds(out) == [("trigger", FOREHEAD)]
    assert out[0].t == pytest.approx(1.0 + 1.0, abs=2 * DT)


def test_busy_too_long_drops_pose_until_released():
    arb = make()
    assert run(arb, {FOREHEAD: True}, 0.0, 4.0, busy=lambda t: t < 3.0) == []
    assert run(arb, {FOREHEAD: True}, 4.0, 6.0) == []  # still held: must be let go first
    out = run(arb, lambda t: {FOREHEAD: t >= 6.6}, 6.0, 8.0)
    assert kinds(out) == [("trigger", FOREHEAD)]


def test_released_while_busy_costs_no_cooldown():
    arb = make()
    run(arb, lambda t: {FOREHEAD: t < 1.0}, 0.0, 2.5, busy=lambda t: t < 2.0)
    assert arb.dwells[FOREHEAD].state == IDLE
    out = run(arb, lambda t: {FOREHEAD: t >= 4.0}, 2.5, 6.0)
    assert kinds(out) == [("trigger", FOREHEAD)]


def test_global_gap_between_different_poses():
    arb = make()
    out = run(arb, lambda t: {EYEBROWS_UP: t < 1.2, FOREHEAD: t >= 1.2}, 0.0, 4.0)
    assert kinds(out) == [("trigger", EYEBROWS_UP), ("trigger", FOREHEAD)]
    assert out[1].t - out[0].t >= 1.0 - 1e-6
    assert out[1].t == pytest.approx(out[0].t + 1.0, abs=DT + 1e-6)  # forehead "on" at 1.9 waits for the gap


def test_same_pose_respects_cooldown():
    arb = make()
    out = run(arb, lambda t: {FOREHEAD: t < 1.5 or 3.0 <= t}, 0.0, 9.0)
    assert kinds(out) == [("trigger", FOREHEAD)]  # 8 s cooldown from ~2.0
    out = run(arb, lambda t: {FOREHEAD: t >= 11.0}, 9.0, 13.0)
    assert kinds(out) == [("trigger", FOREHEAD)]


def test_cover_face_one_frame_truth_triggers_after_settle():
    arb = make()
    k0 = round(1.0 / DT)
    out = run(arb, lambda t: {COVER_FACE: round(t / DT) == k0}, 0.0, 3.0)
    assert kinds(out) == [("trigger", COVER_FACE)]
    assert out[0].t == pytest.approx(1.1, abs=DT + 1e-6)
    assert arb.dwells[COVER_FACE].state == COOLDOWN
    k1 = round(6.0 / DT)
    assert run(arb, lambda t: {COVER_FACE: round(t / DT) == k1}, 3.0, 8.0) == []  # 10 s cooldown


def test_modal_releases_held_cinema():
    arb = make()
    run(arb, {HANDS_UP: True}, 0.0, 3.0)
    arb.modal_active = True
    out = run(arb, {HANDS_UP: True}, 3.0, 5.0)
    assert kinds(out) == [("release", HANDS_UP)]


def test_arbiter_params_from_config():
    arb = make(gestures={"arbiter": {"settle_s": 0.3}, "hands_up": {"t_on": 0.5}})
    out = run(arb, {HANDS_UP: True}, 0.0, 2.0)
    assert out[0].t == pytest.approx(0.5 + 0.3, abs=DT + 1e-6)


def test_status_shape():
    arb = make(grace=3.0)
    run(arb, {FOREHEAD: True}, 0.0, 0.5)
    st = arb.status(0.5)
    assert st["armed"] and st["grace_left"] == pytest.approx(2.5, abs=DT)
    assert set(st["dwell"]) == {HANDS_UP, COVER_FACE, FOREHEAD, "collar", EYEBROWS_UP, CANCEL, THUMBS_UP, "two_thumbs_up", "peace",
                                "point_up", "thumbs_down", "rock"}
    assert st["dwell"][FOREHEAD]["state"] == IDLE
    assert st["suppressed"] == {FOREHEAD: "grace"}
    for key in ("presence", "modal", "busy", "snap_arming", "pending", "holding", "last", "leave_guard_left"):
        assert key in st


def test_peace_sign_plays_the_outro_after_the_leave_guard():
    from zap.gestures.types import PEACE

    arb = make(leave_guard=60.0)
    assert run(arb, {PEACE: True}, 0.0, 3.0) == []  # just armed: not yet
    arb = make()
    out = run(arb, {PEACE: True}, 0.0, 2.0)
    assert kinds(out) == [("trigger", PEACE)]


def test_money_fires_on_top_of_a_showing_meme():
    from zap.gestures.types import TWO_THUMBS

    arb = make()
    out = run(arb, {TWO_THUMBS: True}, 0.0, 2.0, busy=lambda t: True)
    assert kinds(out) == [("trigger", TWO_THUMBS)]
