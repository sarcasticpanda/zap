"""SnapAudioDetector and MicStream on synthetic 48 kHz audio (no microphone)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from zap.devices import audio_in
from zap.devices.audio_in import MicStream
from zap.gestures import snap_audio as sa
from zap.gestures.snap_audio import SnapAudioDetector

R = 48000
T0 = 100.0  # perf_counter time of the first sample


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def bed(rng, sec=3.0, level=3e-4):
    return (rng.standard_normal(int(sec * R)) * level).astype(np.float32)


def band(rng, n, lo, hi):
    sos = butter(4, [lo, hi], btype="bandpass", fs=R, output="sos")
    x = sosfilt(sos, rng.standard_normal(n + 2000))[2000:]
    return x / np.abs(x).max()


def snap(rng, amp=0.3):
    n = int(0.03 * R)
    t = np.arange(n) / R
    env = np.where(t < 2e-4, t / 2e-4, np.exp(-(t - 2e-4) / 0.005))
    return amp * band(rng, n, 1500, 5000) * env


def thud(amp=0.5):
    t = np.arange(int(0.2 * R)) / R
    return amp * np.sin(2 * np.pi * 100 * t) * np.exp(-t / 0.03) * np.minimum(1, t / 5e-4)


def tone(amp=0.2):
    t = np.arange(int(0.3 * R)) / R
    return amp * np.sin(2 * np.pi * 2000 * t) * np.minimum(1, t / 1e-3)


def fricative(rng, amp=0.1):
    n = int(0.2 * R)
    return amp * band(rng, n, 3000, 8000) * np.minimum(1, np.arange(n) / R / 0.03)


def clap(rng, amp=0.5):
    n = int(0.3 * R)
    t = np.arange(n) / R
    y = (rng.standard_normal(n) + 2.5 * band(rng, n, 80, 500)) * np.exp(-t / 0.06) * np.minimum(1, t / 3e-4)
    return amp * y / np.abs(y).max()


def place(sig, events):
    out = sig.copy()
    for t0, ev in events:
        i = int(t0 * R)
        out[i : i + ev.size] += ev[: out.size - i].astype(np.float32)
    return out


def run(sig, det=None, bs=128):
    det = det or SnapAudioDetector()
    out = []
    for i in range(0, sig.size, bs):
        out += det.process(sig[i : i + bs], T0 + i / R)
    return out


def test_snap_accepted_with_precise_onset(rng):
    cs = run(place(bed(rng), [(1.5, snap(rng))]))
    assert len(cs) == 1
    c = cs[0]
    assert c.accepted and c.reason == sa.OK
    assert abs(c.t_onset - (T0 + 1.5)) < 0.006
    assert c.features["snr"] > 10


def test_odd_block_sizes_give_same_result(rng):
    sig = place(bed(rng), [(1.5, snap(rng))])
    cs = run(sig, bs=333)
    assert len(cs) == 1 and cs[0].accepted
    assert abs(cs[0].t_onset - (T0 + 1.5)) < 0.006


@pytest.mark.parametrize(
    ("make", "reasons"),
    [
        (lambda rng: thud(), {sa.LF_HEAVY, sa.NOT_HF, sa.TOO_QUIET}),
        (lambda rng: tone(), {sa.SLOW_DECAY, sa.RINGING, sa.SLOW_ATTACK, sa.NOT_PEAKY}),
        (fricative, {sa.SLOW_DECAY, sa.RINGING, sa.SLOW_ATTACK}),
        (clap, {sa.LF_HEAVY, sa.SLOW_DECAY, sa.RINGING}),
    ],
    ids=["thud", "tone", "fricative", "clap"],
)
def test_non_snaps_rejected(rng, make, reasons):
    cs = run(place(bed(rng), [(1.5, make(rng))]))
    assert cs, "the sound should open a candidate"
    assert not any(c.accepted for c in cs)
    assert cs[0].reason in reasons


def test_thud_is_lf_heavy(rng):
    cs = run(place(bed(rng), [(1.5, thud())]))
    assert cs[0].reason == sa.LF_HEAVY


def test_typing_rejected(rng):
    starts = np.cumsum([0.0, 0.05, 0.06, 0.045, 0.07, 0.055])
    clicks = [(1.5 + s, snap(rng, 0.025)) for s in starts]
    cs = run(place(bed(rng), clicks))
    assert cs
    assert not any(c.accepted for c in cs)
    assert {c.reason for c in cs} <= {sa.SLOW_DECAY, sa.RINGING, sa.NOISY_BEFORE}


def test_triple_snap(rng):
    cs = run(place(bed(rng), [(1.5, snap(rng)), (1.8, snap(rng)), (2.1, snap(rng))]))
    ok = [c.t_onset - T0 for c in cs if c.accepted]
    assert len(ok) == 3
    assert np.allclose(ok, [1.5, 1.8, 2.1], atol=0.006)


def test_self_gate(rng):
    det = SnapAudioDetector()
    det.gate(T0 + 1.6)
    cs = run(place(bed(rng), [(1.5, snap(rng)), (2.2, snap(rng))]), det)
    assert [(c.accepted, c.reason) for c in cs] == [(False, sa.OWN_SOUND), (True, sa.OK)]


def test_quiet_sounds_never_open_candidates(rng):
    cs = run(place(bed(rng, level=1e-5), [(1.5, snap(rng, amp=2e-4))]))
    assert cs == []


def test_noise_floor_tracks_bed(rng):
    det = SnapAudioDetector()
    run(bed(rng, sec=2.0, level=1e-3), det)
    assert det.ready
    assert -75 < det.noise_floor_dbfs < -55


def test_sensitivity_scales_thresholds():
    strict, mid, loose = (sa.Thresholds.for_sensitivity(s) for s in (1, 3, 5))
    assert mid == sa.Thresholds()
    assert strict.snr > mid.snr > loose.snr
    assert strict.lf_ratio < mid.lf_ratio < loose.lf_ratio
    assert strict.decay_hf < mid.decay_hf < loose.decay_hf
    det = SnapAudioDetector(sensitivity=9)
    assert det.sensitivity == 5


# ------------------------------------------------------------ MicStream (no device)
def test_block_start_uses_adc_time():
    mic = MicStream(SnapAudioDetector())
    ti = SimpleNamespace(inputBufferAdcTime=10.000, currentTime=10.012)
    assert mic.block_start(50.0, 128, ti) == pytest.approx(50.0 - 0.012 - 128 / R)
    mic._latency = 0.01
    ti0 = SimpleNamespace(inputBufferAdcTime=0.0, currentTime=0.0)
    assert mic.block_start(50.0, 128, ti0) == pytest.approx(50.0 - 128 / R - 0.01)


def test_callback_queue_and_analysis(rng):
    mic = MicStream(SnapAudioDetector())
    sig = place(bed(rng), [(1.5, snap(rng))])
    status = SimpleNamespace(input_overflow=False)
    for i in range(0, sig.size, 128):
        ti = SimpleNamespace(inputBufferAdcTime=0.0, currentTime=0.0)
        mic._callback(sig[i : i + 128, None], 128, ti, status)
        # Callback timestamps (fallback path) are block-relative; rewrite them as contiguous.
        mic._blocks[-1] = (mic._blocks[-1][0], T0 + i / R)
    for block, t0 in mic._batches():
        mic._analyse(block, t0)
    cs = mic.drain()
    assert [c.accepted for c in cs] == [True]
    assert abs(cs[0].t_onset - (T0 + 1.5)) < 0.006
    assert mic.drain() == []
    assert -80 < mic.level < 0


def test_batches_split_on_gaps():
    mic = MicStream(SnapAudioDetector())
    b = np.zeros(128, np.float32)
    for t in (0.0, 128 / R, 2 * 128 / R, 1.0):
        mic._blocks.append((b, t))
    runs = mic._batches()
    assert [(r[0].size, r[1]) for r in runs] == [(384, 0.0), (128, 1.0)]


def test_start_failure_sets_error(monkeypatch):
    def boom(name=None):
        raise LookupError("no microphone found")

    monkeypatch.setattr(audio_in, "find_input_device", boom)
    mic = MicStream(SnapAudioDetector(), device="nothing")
    assert mic.start() is False
    assert mic.error and "no microphone" in mic.error
    assert not mic.running
    mic.stop()
