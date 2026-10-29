"""Procedural sound effects, so every effect has a sound without shipping audio.

Meme sounds are copyrighted and never bundled; users may drop their own WAVs in
``assets/sfx``. When a key has no file, :func:`synth` makes a stand-in from
noise, sines and wavetables. Everything is vectorized numpy/scipy (no
per-sample loops), seeded per key so a sound is identical on every run, and
cached because effects ask for the same few keys over and over.

All sounds are mono float32 at 48 kHz, peaking near 0.5, at most 2.5 s except the
"we'll be right back" music.
"""

from __future__ import annotations

import logging
import zlib
from collections.abc import Callable
from functools import lru_cache

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 48000
PEAK = 0.5
SOFT_PEAK = 0.3  # the fallback blip for unknown keys is deliberately quieter
MAX_SECONDS = 2.5
SEED = 0x5A9  # change to re-roll every sound

KEYS = ("presence", "reappear", "money", "cinema", "sad", "cutaway", "outro", "charge", "brb")

_WT = 2048  # wavetable length (one cycle)


# ---------------------------------------------------------------- building blocks
def _signal():
    """scipy.signal, imported on first use: it takes seconds to import, and
    importing this module (e.g. for KEYS) should not."""
    from scipy import signal

    return signal


def _time(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE


def _sos(kind: str, freq, order: int = 4) -> np.ndarray:
    return _signal().butter(order, freq, btype=kind, fs=SAMPLE_RATE, output="sos")


def _filt(x: np.ndarray, kind: str, freq, order: int = 4) -> np.ndarray:
    return _signal().sosfilt(_sos(kind, freq, order), x)


def _sweep_noise(rng: np.random.Generator, seconds: float, f0: float, f1: float, octaves: float = 1.0) -> np.ndarray:
    """Noise whose spectral centre glides log-linearly from f0 to f1 (built in the STFT domain)."""
    nper, hop = 1024, 256
    n = int(seconds * SAMPLE_RATE)
    frames = n // hop + 4
    t = np.linspace(0.0, 1.0, frames)
    fc = f0 * (f1 / f0) ** t  # (frames,)
    freqs = np.fft.rfftfreq(nper, 1 / SAMPLE_RATE)[:, None]  # (bins, 1)
    with np.errstate(divide="ignore"):
        dist = np.log2(np.maximum(freqs, 1.0) / fc[None, :]) / octaves
    mag = np.exp(-0.5 * dist**2)
    phase = rng.uniform(0, 2 * np.pi, mag.shape)
    _, x = _signal().istft(mag * np.exp(1j * phase), fs=SAMPLE_RATE, nperseg=nper, noverlap=nper - hop)
    x = x[:n]
    return x / (np.abs(x).max() + 1e-9)


def _grains(rng: np.random.Generator, n: int, count: int, weights: np.ndarray, grain_ms: float = 3.0,
            rise_ms: float = 0.0) -> np.ndarray:
    """Sparse dust/crackle: `count` tiny noise grains at times drawn from `weights` (length n)."""
    p = weights / weights.sum()
    idx = rng.choice(n, size=count, p=p)
    imp = np.zeros(n)
    np.add.at(imp, idx, rng.uniform(0.2, 1.0, count) * rng.choice((-1.0, 1.0), count))
    k = np.arange(int(grain_ms * 1e-3 * SAMPLE_RATE)) / SAMPLE_RATE
    kernel = rng.standard_normal(k.size) * np.exp(-k / (grain_ms * 3e-4))
    if rise_ms > 0:
        kernel *= 1 - np.exp(-k / (rise_ms * 1e-3))
    return _signal().fftconvolve(imp, kernel)[:n]


def _wavetable(rolloff: float, harmonics: int) -> np.ndarray:
    """One cycle of a band-limited tone with harmonic amplitudes 1/n**rolloff."""
    ph = np.arange(_WT) / _WT * 2 * np.pi
    n = np.arange(1, harmonics + 1)[:, None]
    table = (np.sin(n * ph[None, :]) / n**rolloff).sum(axis=0)
    return table / np.abs(table).max()


def _osc(table: np.ndarray, freq) -> np.ndarray:
    """Read a wavetable at a (possibly time-varying) frequency in Hz."""
    phase = np.cumsum(np.broadcast_to(freq, np.shape(freq))) / SAMPLE_RATE
    return np.take(table, (phase * _WT).astype(np.int64) & (_WT - 1))


def _reverb(x: np.ndarray, seconds: float, wet: float, rng: np.random.Generator) -> np.ndarray:
    t = _time(seconds)
    ir = rng.standard_normal(t.size) * np.exp(-t / (seconds / 5))
    ir = _filt(ir, "lowpass", 5000, 2)
    tail = _signal().fftconvolve(x, ir)[: x.size]
    tail *= np.abs(x).max() / (np.abs(tail).max() + 1e-9)
    return x + wet * tail


def _place(out: np.ndarray, x: np.ndarray, at_s: float) -> None:
    i = int(at_s * SAMPLE_RATE)
    j = min(out.size, i + x.size)
    if j > i:
        out[i:j] += x[: j - i]


def _bell(seconds: float, f: float, tau: float) -> np.ndarray:
    t = _time(seconds)
    ratios = np.array([1.0, 2.0, 2.76, 5.40])[:, None]
    amps = np.array([1.0, 0.5, 0.35, 0.2])[:, None]
    decays = tau / np.array([1.0, 1.5, 2.0, 3.0])[:, None]
    return (amps * np.sin(2 * np.pi * f * ratios * t) * np.exp(-t / decays)).sum(axis=0)


# ---------------------------------------------------------------- the sounds
def _presence(rng: np.random.Generator) -> np.ndarray:
    """Thanos vanish: a rising dusty whoosh."""
    dur = 1.6
    t = _time(dur)
    u = t / dur
    env = np.sin(np.pi * u**0.75) ** 1.5
    whoosh = np.tanh(2.0 * _sweep_noise(rng, dur, 350, 5200, 0.6)) * env
    dust = _filt(_grains(rng, t.size, 450, 0.05 + u**2), "highpass", 1800, 2) * env
    return whoosh + 0.4 * dust / (np.abs(dust).max() + 1e-9)


def _reappear(rng: np.random.Generator) -> np.ndarray:
    """Reassembling: a dusty whoosh played backwards (slow swell, abrupt stop)."""
    dur = 1.2
    t = _time(dur)
    env = (1 - np.exp(-t / 0.02)) * np.exp(-t / 0.33)
    whoosh = np.tanh(2.0 * _sweep_noise(rng, dur, 5000, 400, 0.6)) * env
    dust = _filt(_grains(rng, t.size, 300, np.exp(-t / 0.3)), "highpass", 1800, 2)
    x = whoosh + 0.35 * dust / (np.abs(dust).max() + 1e-9) * env
    return x[::-1].copy()


def _money(rng: np.random.Generator) -> np.ndarray:
    """Cash register: a 'ka' click, a two-tone 'ching', then a shimmer of coins."""
    out = np.zeros(int(1.7 * SAMPLE_RATE))
    k = _time(0.05)
    ka = _filt(rng.standard_normal(k.size), "bandpass", (1800, 6500)) * np.exp(-k / 0.006)
    ka += 0.6 * np.sin(2 * np.pi * 190 * k) * np.exp(-k / 0.02)
    _place(out, ka, 0.0)
    _place(out, 0.55 * _bell(0.6, 1975.5, 0.25), 0.07)
    _place(out, 0.7 * _bell(1.4, 2637.0, 0.55), 0.17)
    count, length = 34, int(0.08 * SAMPLE_RATE)
    starts = (rng.uniform(0.22, 1.35, count) * SAMPLE_RATE).astype(np.int64)
    freqs = rng.uniform(3500, 8000, count)[:, None]
    amps = (0.25 * np.exp(-(starts - starts.min()) / (0.6 * SAMPLE_RATE)))[:, None]
    tl = np.arange(length)[None, :] / SAMPLE_RATE
    pings = amps * np.sin(2 * np.pi * freqs * tl) * np.exp(-tl / 0.022)
    idx = np.minimum(starts[:, None] + np.arange(length)[None, :], out.size - 1)
    np.add.at(out, idx, pings)
    return out


def _cinema(rng: np.random.Generator) -> np.ndarray:
    """Absolute Cinema: a deep boom under a bright orchestral chord hit."""
    dur = 2.5
    t = _time(dur)
    f = 38 + 52 * np.exp(-t / 0.12)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SAMPLE_RATE) * np.exp(-t / 0.75)
    thump = _filt(rng.standard_normal(t.size), "lowpass", 160) * np.exp(-t / 0.06)
    low = np.tanh(2.2 * (boom + 0.8 * thump / (np.abs(thump).max() + 1e-9)))
    table = _wavetable(1.15, 14)
    notes = np.array([261.63, 392.0, 523.25, 659.26, 783.99])
    detune = 2 ** (np.array([-4.0, 4.0]) / 1200)
    env = (1 - np.exp(-t / 0.006)) * np.exp(-t / 0.55)
    chord = sum(_osc(table, np.full(t.size, f0 * d)) for f0 in notes for d in detune)
    chord = chord / (np.abs(chord).max() + 1e-9) * env
    x = 0.9 * low + 0.6 * _filt(chord, "lowpass", 7000, 2)
    return _reverb(x, 1.2, 0.3, rng)


def _sad(rng: np.random.Generator) -> np.ndarray:
    """Sad trombone: wah, wah, wah, waaaah (descending semitones, last one wobbles)."""
    notes = (196.0, 185.0, 174.61, 164.81)
    onsets = (0.0, 0.45, 0.9, 1.35)
    lengths = (0.42, 0.42, 0.42, 1.1)
    n = int(2.5 * SAMPLE_RATE)
    freq = np.full(n, notes[0])
    amp = np.zeros(n)
    bright = np.zeros(n)
    for f0, on, ln in zip(notes, onsets, lengths, strict=True):  # four notes, vectorized inside
        i0, i1 = int(on * SAMPLE_RATE), int((on + ln) * SAMPLE_RATE)
        tn = np.arange(i1 - i0) / SAMPLE_RATE
        cents = 35 * np.exp(-tn / 0.05)
        if ln > 0.5:
            vib = np.clip((tn - 0.3) / 0.2, 0, 1) * 28 * np.sin(2 * np.pi * 5.5 * tn)
            cents = cents + vib - 60 * np.clip((tn - 0.7) / 0.4, 0, 1)
        freq[i0:] = f0
        freq[i0:i1] = f0 * 2 ** (cents / 1200)
        amp[i0:i1] = (1 - np.exp(-tn / 0.025)) * np.clip((ln - tn) / 0.07, 0, 1)
        bright[i0:i1] = np.sin(np.pi * np.clip(tn / (0.8 * ln), 0, 1)) ** 0.8
    dark = _osc(_wavetable(2.6, 16), freq)
    brassy = _osc(_wavetable(1.2, 16), freq)
    x = (dark * (1 - bright) + brassy * bright) * amp
    return _reverb(_filt(x, "lowpass", 3500, 2), 0.6, 0.15, rng)


def _cutaway(rng: np.random.Generator) -> np.ndarray:
    """A vine-boom-style low thud for meme cutaways."""
    dur = 1.3
    t = _time(dur)
    f = 46 + 70 * np.exp(-t / 0.07)
    ph = 2 * np.pi * np.cumsum(f) / SAMPLE_RATE
    body = (np.sin(ph) + 0.3 * np.sin(2 * ph)) * (1 - np.exp(-t / 0.003)) * np.exp(-t / 0.42)
    body = _filt(np.tanh(3.0 * body), "lowpass", 900, 2)
    k = _time(0.012)
    click = _filt(rng.standard_normal(k.size), "bandpass", (900, 4000), 2) * np.exp(-k / 0.002)
    x = body.copy()
    _place(x, 0.25 * click, 0.0)
    return _reverb(x, 0.6, 0.25, rng)


def _outro(rng: np.random.Generator) -> np.ndarray:
    """A short plucked four-note 'to be continued' jingle (original, not the song)."""
    notes = (392.0, 493.88, 587.33, 783.99)
    onsets = (0.0, 0.15, 0.30, 0.45)
    lengths = (0.3, 0.3, 0.3, 1.25)
    out = np.zeros(int(1.75 * SAMPLE_RATE))
    h = np.arange(1, 9)[:, None]
    for f0, on, ln in zip(notes, onsets, lengths, strict=True):
        t = _time(ln)[None, :]
        partials = np.sin(2 * np.pi * f0 * h * t) / h**1.1 * np.exp(-t * (2.5 + 2.0 * h))
        note = partials.sum(axis=0) * (1 - np.exp(-t[0] / 0.003)) * np.clip((ln - t[0]) / 0.03, 0, 1)
        _place(out, note, on)
    tb = _time(1.25)
    bass = np.sin(2 * np.pi * 98.0 * tb) * np.exp(-tb / 0.4) * (1 - np.exp(-tb / 0.004))
    _place(out, 0.6 * bass, 0.45)
    return _reverb(out, 0.7, 0.2, rng)


def _brb(rng: np.random.Generator) -> np.ndarray:
    """'We'll be right back': a jaunty bumper (walking bass, brass stabs, a held last chord).
    An original stand-in, not the meme's song."""
    e = 60 / 132 / 2  # an eighth note at 132 bpm
    out = np.zeros(int(4.4 * SAMPLE_RATE))
    bass = (65.41, 82.41, 98.0, 110.0, 87.31, 110.0, 130.81, 146.83, 98.0, 123.47, 146.83, 164.81)
    for i, f0 in enumerate(bass):  # C E G A | F A C D | G B D E, plucked
        t = _time(0.95 * e)
        note = (np.sin(2 * np.pi * f0 * t) + 0.35 * np.sin(4 * np.pi * f0 * t)) * np.exp(-t / 0.14)
        _place(out, 0.8 * note * (1 - np.exp(-t / 0.004)), i * e)
    brass = _wavetable(1.2, 16)
    chords = ((0, (261.63, 329.63, 392.0, 440.0)), (3, (261.63, 329.63, 392.0, 440.0)),
              (4, (349.23, 440.0, 523.25, 587.33)), (7, (349.23, 440.0, 523.25, 587.33)),
              (8, (392.0, 493.88, 587.33, 659.26)), (10, (392.0, 493.88, 587.33, 698.46)))
    for step, notes in chords:  # short stabs on the off-beats
        t = _time(0.2)
        env = (1 - np.exp(-t / 0.008)) * np.exp(-t / 0.07)
        stab = sum(_osc(brass, np.full(t.size, f)) for f in notes) / len(notes)
        _place(out, 0.55 * _filt(stab, "lowpass", 4500, 2) * env, step * e)
    t = _time(1.6)  # the last chord, a C 6/9 with a little vibrato, swelling then fading
    vib = 2 ** (8 * np.sin(2 * np.pi * 5.0 * t) * np.clip((t - 0.3) / 0.4, 0, 1) / 1200)
    last = sum(_osc(brass, f * vib) for f in (261.63, 329.63, 392.0, 440.0, 587.33)) / 5
    env = (1 - np.exp(-t / 0.03)) * np.clip((1.6 - t) / 0.5, 0, 1)
    _place(out, 0.7 * _filt(last, "lowpass", 5000, 2) * env, 12 * e)
    tb = _time(1.6)
    _place(out, 0.7 * np.sin(2 * np.pi * 65.41 * tb) * np.exp(-tb / 0.6), 12 * e)
    tc = _time(0.5)  # a brushed cymbal swell into the last chord
    cym = _filt(rng.standard_normal(tc.size), "highpass", 6000, 2) * (tc / 0.5) ** 2
    _place(out, 0.25 * cym, 12 * e - 0.5)
    return _reverb(out, 0.8, 0.18, rng)


def _charge(rng: np.random.Generator) -> np.ndarray:
    """Soft ember crackle for the snap pre-roll. Low-passed with rounded grains so the
    microphone snap detector does not mistake it for a snap."""
    dur = 0.9
    t = _time(dur)
    env = np.clip(t / 0.3, 0, 1) * np.clip((dur - t) / 0.3, 0, 1)
    crackle = _filt(_grains(rng, t.size, 70, np.ones(t.size), grain_ms=6.0, rise_ms=1.5), "lowpass", 1400, 4)
    bed = _filt(rng.standard_normal(t.size), "bandpass", (180, 800), 2)
    crackle /= np.abs(crackle).max() + 1e-9
    bed /= np.abs(bed).max() + 1e-9
    return (0.8 * crackle + 0.35 * bed) * env


def _blip(rng: np.random.Generator) -> np.ndarray:
    t = _time(0.14)
    return np.sin(2 * np.pi * 880 * t) * (1 - np.exp(-t / 0.004)) * np.exp(-t / 0.035)


_MAKERS: dict[str, Callable[[np.random.Generator], np.ndarray]] = {
    "presence": _presence,
    "reappear": _reappear,
    "money": _money,
    "cinema": _cinema,
    "sad": _sad,
    "cutaway": _cutaway,
    "outro": _outro,
    "charge": _charge,
    "brb": _brb,
}
LONGER = {"brb": 4.4}  # music, not a sound effect: allowed past MAX_SECONDS


def _finish(x: np.ndarray, peak: float, seconds: float = MAX_SECONDS) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)[: int(seconds * SAMPLE_RATE)]
    x = x - x.mean()
    fi, fo = int(0.003 * SAMPLE_RATE), int(0.012 * SAMPLE_RATE)
    x[:fi] *= np.linspace(0, 1, fi)
    x[-fo:] *= np.linspace(1, 0, fo)
    x *= peak / (np.abs(x).max() + 1e-12)
    out = x.astype(np.float32)
    out.flags.writeable = False  # shared through the cache
    return out


@lru_cache(maxsize=64)
def synth(key: str) -> np.ndarray:
    """Mono float32 at 48 kHz for a sound key; unknown keys get a short soft blip.

    The returned array is cached and read-only.
    """
    rng = np.random.default_rng(SEED ^ zlib.crc32(key.encode("utf-8")))
    maker = _MAKERS.get(key)
    if maker is None:
        log.debug("no procedural sound for '%s'; using a blip", key)
        return _finish(_blip(rng), SOFT_PEAK)
    return _finish(maker(rng), PEAK, LONGER.get(key, MAX_SECONDS))


def prewarm(keys=KEYS) -> None:
    """Generate sounds ahead of time (each takes tens of milliseconds the first time)."""
    for k in keys:
        synth(k)
