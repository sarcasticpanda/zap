from __future__ import annotations

import itertools

import pytest

from zap.gestures.dwell import ACTIVE, ARMING, COOLDOWN, IDLE, WAIT_RELEASE, Dwell, DwellParams

DT = 1 / 30


def run(d: Dwell, truth_at, t0: float, t1: float, dt: float = DT) -> list[tuple[float, str]]:
    """Feed frames at k*dt for t in [t0, t1); returns (t, event) for every non-None event."""
    out = []
    for k in range(round(t0 / dt), round(t1 / dt)):
        t = k * dt
        ev = d.update(bool(truth_at(t)), t)
        if ev:
            out.append((t, ev))
    return out


def P(**kw) -> DwellParams:
    base = {"t_on": 0.4, "t_off": 0.6, "min_show": 0.0, "cooldown": 4.0}
    base.update(kw)
    return DwellParams(**base)


def test_on_after_t_on_of_continuous_truth():
    d = Dwell(P())
    ev = run(d, lambda t: True, 0.0, 1.0)
    assert ev[0][1] == "on"
    assert ev[0][0] == pytest.approx(0.4, abs=1e-6)
    assert d.state == ACTIVE and d.progress == 1.0
    assert len(ev) == 1


def test_brief_pose_never_fires():
    d = Dwell(P())
    assert run(d, lambda t: t < 0.3, 0.0, 2.0) == []
    assert d.state == IDLE


def test_short_dropout_does_not_reset_arming():
    d = Dwell(P())
    # two dropped frames (67 ms) inside the arming window
    ev = run(d, lambda t: not (0.19 < t < 0.25), 0.0, 1.0)
    assert ev[0][1] == "on"
    assert ev[0][0] < 0.5  # not restarted from the re-acquisition at 0.267


def test_long_dropout_resets_arming():
    d = Dwell(P())
    ev = run(d, lambda t: not (0.19 < t < 0.4), 0.0, 1.5)  # 200 ms gone > dropout
    assert ev[0][1] == "on"
    assert ev[0][0] >= 0.4 + 0.4 - 1e-6  # full t_on again after coming back at ~0.4


def test_flicker_below_ratio_never_fires():
    d = Dwell(P())
    k = lambda t: round(t / DT) % 2 == 0
    assert run(d, k, 0.0, 3.0) == []
    assert d.state == ARMING


def test_mostly_true_flicker_fires():
    d = Dwell(P())
    ev = run(d, lambda t: round(t / DT) % 6 != 5, 0.0, 2.0)  # 83% true
    assert ev and ev[0][1] == "on"


def test_off_after_t_off():
    d = Dwell(P())
    ev = run(d, lambda t: t < 1.0, 0.0, 3.0)
    assert [e for _, e in ev] == ["on", "off"]
    assert ev[1][0] == pytest.approx(1.0 + 0.6, abs=DT + 1e-6)
    assert d.state == COOLDOWN


def test_short_loss_while_active_is_hysteresis():
    d = Dwell(P())
    ev = run(d, lambda t: not (1.0 < t < 1.4), 0.0, 3.0)  # 0.4 s gone < t_off
    assert [e for _, e in ev] == ["on"]
    assert d.state == ACTIVE


def test_min_show_delays_off():
    d = Dwell(P(min_show=2.5))
    ev = run(d, lambda t: t < 0.6, 0.0, 4.0)
    assert [e for _, e in ev] == ["on", "off"]
    assert ev[1][0] == pytest.approx(0.4 + 2.5, abs=DT + 1e-6)


def test_cooldown_then_wait_release():
    d = Dwell(P(cooldown=2.0, rearm=0.4))
    ev = run(d, lambda t: t < 1.0 or 1.8 < t, 0.0, 10.0)  # comes back during cooldown and stays
    assert [e for _, e in ev] == ["on", "off"]  # held through the cooldown: never re-fires
    assert d.state == WAIT_RELEASE


def test_rearms_after_release():
    d = Dwell(P(cooldown=1.0, rearm=0.4))
    # on at 0.4, off at 1.6, cooldown to 2.6; released (false) 2.0..3.2 then held again
    ev = run(d, lambda t: t < 1.0 or 3.2 <= t, 0.0, 5.0)
    assert [e for _, e in ev] == ["on", "off", "on"]
    assert ev[2][0] == pytest.approx(3.2 + 0.4, abs=DT + 1e-6)


def test_release_too_short_for_rearm():
    d = Dwell(P(cooldown=1.0, rearm=0.4))
    # back at 2.0 (in cooldown); released only 0.2 s at 3.0
    ev = run(d, lambda t: t < 1.0 or (2.0 < t < 3.0) or t >= 3.2, 0.0, 5.0)
    assert [e for _, e in ev] == ["on", "off"]


def test_progress_ramps_while_arming():
    d = Dwell(P(t_on=1.0))
    seen = []
    for k in range(25):
        d.update(True, k * DT)
        seen.append(d.progress)
    assert seen[0] == 0.0
    assert all(b >= a for a, b in itertools.pairwise(seen))
    assert 0.7 < seen[-1] < 0.9
    assert d.info(25 * DT)["state"] == ARMING


@pytest.mark.parametrize("fps", [10, 15, 30, 60])
def test_timing_independent_of_frame_rate(fps):
    d = Dwell(P())
    ev = run(d, lambda t: True, 0.0, 1.0, dt=1 / fps)
    assert ev[0][0] == pytest.approx(0.4, abs=1 / fps + 1e-6)


def test_zero_hold_fires_immediately_and_off_next_frame():
    d = Dwell(P(t_on=0.0, t_off=0.0, cooldown=10.0))
    assert d.update(True, 1.0) == "on"
    assert d.update(False, 1.0 + DT) == "off"
    assert d.state == COOLDOWN
    assert d.cooldown_left(2.0) == pytest.approx(10.0 - (1.0 - DT))


def test_t_on_override_per_frame():
    d = Dwell(P(t_on=0.5))
    ev = [(k * DT, d.update(True, k * DT, t_on=1.0)) for k in range(45)]
    on = [t for t, e in ev if e == "on"]
    assert on and on[0] == pytest.approx(1.0, abs=1e-6)


def test_cancel_requires_release():
    d = Dwell(P(rearm=0.3))
    run(d, lambda t: True, 0.0, 0.5)
    d.cancel(0.5)
    assert d.state == WAIT_RELEASE
    assert run(d, lambda t: True, 0.5, 2.0) == []
    ev = run(d, lambda t: t >= 2.4, 2.0, 3.5)
    assert [e for _, e in ev] == ["on"]


def test_reset_goes_idle():
    d = Dwell(P())
    run(d, lambda t: True, 0.0, 0.3)
    d.reset()
    assert d.state == IDLE and d.progress == 0.0
