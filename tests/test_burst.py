"""BurstCounter grouping of snap onsets."""

from __future__ import annotations

from zap.gestures.snap import BurstCounter
from zap.gestures.types import Intent, SnapBurst


def feed(bc, onsets, until, step=1 / 30):
    """Add onsets at their times and tick every frame; returns (bursts, markers, charges)."""
    bursts, markers, charges = [], [], []
    todo = sorted(onsets)
    k = 0
    while k * step <= until:
        now = k * step
        while todo and todo[0] <= now:
            m = bc.add(todo.pop(0), (10.0, 20.0))
            if m is not None:
                charges.append(m)
        b, mk = bc.tick(now)
        bursts += b
        markers += mk
        k += 1
    return bursts, markers, charges


def test_single_double_triple():
    for n in (1, 2, 3):
        onsets = [1.0 + 0.3 * i for i in range(n)]
        bursts, markers, charges = feed(BurstCounter(), onsets, 4.0)
        assert [b.count for b in bursts] == [n]
        assert bursts[0].t_first == 1.0 and bursts[0].t_last == onsets[-1]
        assert bursts[0].where == (10.0, 20.0)
        assert markers == []
        assert len(charges) == 1 and charges[0].kind == "charge" and charges[0].t == 1.0


def test_burst_completes_after_gap():
    bc = BurstCounter()
    assert bc.add(1.0) == Intent("charge", t=1.0)
    assert bc.tick(1.6) == ([], [])
    assert bc.in_burst and bc.count == 1
    bursts, _ = bc.tick(1.66)
    assert bursts == [SnapBurst(1, 1.0, 1.0, None)]
    assert not bc.in_burst


def test_more_than_max_discarded():
    bursts, markers, charges = feed(BurstCounter(), [1.0, 1.3, 1.6, 1.9], 4.0)
    assert bursts == []
    assert [m.kind for m in markers] == ["discharge"]
    assert len(charges) == 1


def test_refractory_merges_echo():
    bursts, _, _ = feed(BurstCounter(), [1.0, 1.1], 3.0)
    assert [b.count for b in bursts] == [1]


def test_lockout_after_dispatch():
    bc = BurstCounter()
    bc.add(1.0)
    bursts, _ = bc.tick(1.7)
    assert len(bursts) == 1
    assert bc.add(2.3) is None  # within 0.8 s of the dispatch
    assert not bc.in_burst
    assert bc.add(2.6) is not None


def test_slow_snaps_are_separate():
    # 0.7 s apart is beyond the gap; the second one falls in the lockout.
    bursts, _, _ = feed(BurstCounter(), [1.0, 1.7], 4.0)
    assert [b.count for b in bursts] == [1]
    bursts, _, _ = feed(BurstCounter(), [1.0, 2.6], 4.0)
    assert [b.count for b in bursts] == [1, 1]


def test_pending_onset_holds_burst_open():
    bc = BurstCounter()
    bc.add(1.0)
    assert bc.tick(1.7, None, 1.5) == ([], [])  # an undecided onset at 1.5 could still join
    bc.add(1.5)
    assert bc.tick(2.1, None) == ([], [])
    bursts, _ = bc.tick(2.2, 3.0)  # a pending onset beyond the gap does not hold
    assert [b.count for b in bursts] == [2]


def test_clamp_for_vision_only():
    bursts, markers, _ = feed(BurstCounter(max_count=1, clamp=True), [1.0, 1.4], 3.0)
    assert [b.count for b in bursts] == [1]
    assert markers == []
