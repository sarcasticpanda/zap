"""Procedural sounds: shape, level, length, determinism. Nothing is played."""

from __future__ import annotations

import numpy as np
import pytest

from zap.render import synth_sfx
from zap.render.synth_sfx import KEYS, MAX_SECONDS, PEAK, SAMPLE_RATE, synth

EXPECTED_SECONDS = {  # rough lengths from the spec; generous bounds
    "presence": (1.3, 2.0),
    "reappear": (0.8, 1.6),
    "money": (1.0, 2.0),
    "cinema": (1.5, 2.5),
    "sad": (1.8, 2.5),
    "cutaway": (0.8, 1.6),
    "tbc": (1.2, 2.0),
    "charge": (0.5, 1.2),
    "brb": (3.8, 4.4),  # music: longer than a sound effect
}


@pytest.mark.parametrize("key", KEYS)
def test_sound_format_and_level(key):
    x = synth(key)
    assert x.dtype == np.float32 and x.ndim == 1
    lo, hi = EXPECTED_SECONDS[key]
    assert lo * SAMPLE_RATE <= x.size <= min(hi, synth_sfx.LONGER.get(key, MAX_SECONDS)) * SAMPLE_RATE
    assert np.isfinite(x).all()
    assert np.abs(x).max() == pytest.approx(PEAK, abs=0.01)
    assert abs(float(x.mean())) < 0.01  # no DC offset
    assert abs(float(x[0])) < 0.05 and abs(float(x[-1])) < 0.05  # faded ends: no clicks
    assert float(np.sqrt(np.mean(x**2))) > 0.02  # actually audible, not a near-silent tail


def test_sounds_differ():
    rms = {k: synth(k)[:SAMPLE_RATE // 2] for k in KEYS}
    for i, a in enumerate(KEYS):
        for b in KEYS[i + 1 :]:
            n = min(rms[a].size, rms[b].size)
            assert not np.allclose(rms[a][:n], rms[b][:n])


def test_deterministic_across_cache_clears():
    a = {k: synth(k).copy() for k in KEYS}
    synth_sfx.synth.cache_clear()
    for k in KEYS:
        np.testing.assert_array_equal(synth(k), a[k])


def test_cached_and_read_only():
    x = synth("money")
    assert synth("money") is x
    with pytest.raises(ValueError):
        x[0] = 1.0


def test_unknown_key_is_short_soft_blip():
    x = synth("definitely-not-a-sound")
    assert x.dtype == np.float32 and x.size < 0.3 * SAMPLE_RATE
    assert 0.1 < np.abs(x).max() < PEAK


def test_presence_rises_and_reappear_ends_abruptly():
    p = synth("presence").astype(np.float64)
    third = p.size // 3
    assert np.abs(p[third : 2 * third]).mean() > np.abs(p[:third // 3]).mean()
    r = synth("reappear").astype(np.float64)
    tenth = r.size // 10
    assert np.abs(r[-2 * tenth : -tenth]).mean() > np.abs(r[:tenth]).mean()  # swells toward the end


def test_charge_is_low_passed():
    """The snap detector listens for bright transients; the pre-roll crackle must not look like one."""
    x = synth("charge").astype(np.float64)
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, 1 / SAMPLE_RATE)
    assert spec[f > 3000].sum() / spec.sum() < 0.02
