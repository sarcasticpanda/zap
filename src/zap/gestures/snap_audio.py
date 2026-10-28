"""Finger-snap detection from the microphone.

A snap is a <3 ms acoustic transient with most of its energy at 1.5-4.5 kHz.
A 30 fps camera only sees the hand before and after it, so the microphone
decides WHEN a snap happened and vision (``snap.py``) only has to agree.

The detector runs on short hops (128 samples = 2.7 ms at 48 kHz) of three
signals: an HF band (1.5-8 kHz), an LF band (<600 Hz) and the full band. A
candidate opens on a sudden HF rise over an adaptive noise floor and is judged
150 ms later on its shape: snaps are loud, HF-dominated, start abruptly from
quiet and die within ~30 ms. Claps and knocks carry strong LF, tones and speech
decay slowly, fricatives ramp up and typing never starts from quiet.

Every judged candidate is returned, accepted or not, with the reason, so the
HUD can show why a sound was ignored.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfilt

log = logging.getLogger(__name__)

MIN_NF = 1e-10  # noise floor never drops below this (digital silence)
MIN_TRIGGER = 1e-6  # ~-60 dBFS hop energy: quieter sounds never open a candidate
RISE = 2.0  # a candidate needs Eh to jump this much over the previous ~8 ms
TINY = 1e-20

# Reject reasons (shown on the HUD).
OK = "ok"
OWN_SOUND = "own sound"
TOO_QUIET = "too quiet"
LF_HEAVY = "LF heavy"
NOT_HF = "not HF"
NOISY_BEFORE = "noisy before"
SLOW_DECAY = "slow decay"
RINGING = "ringing"
SLOW_ATTACK = "slow attack"
NOT_PEAKY = "not peaky"


@dataclass
class AudioCandidate:
    t_onset: float  # perf_counter seconds of the transient's first hop
    features: dict[str, float] = field(default_factory=dict)
    accepted: bool = False
    reason: str = ""


@dataclass(frozen=True)
class Thresholds:
    """Shape limits for a snap. Defaults are sensitivity 3."""

    snr: float = 10.0  # min peak Eh / noise floor
    lf_ratio: float = 2.0  # max sum El / sum Eh over the first 10 ms
    hf_share: float = 0.10  # min Eh / Ef at the peak
    quiet_before: float = 1.0 / 3.0  # max (loudest hop in the 200 ms before) / peak
    decay_hf: float = 0.25  # max Eh in [30, 150] ms / peak
    decay_full: float = 0.40  # same for the full band
    attack_hops: int = 3  # max hops from Eh > peak/3 to the peak
    # Min HF crest factor over 10 ms around the peak. 3, not 4: noise-like snaps
    # measure 3.1-5.5 here, while tonal blips (the case this catches) are below 2.
    crest: float = 3.0

    @classmethod
    def for_sensitivity(cls, sensitivity: int) -> Thresholds:
        """1 = strictest, 5 = loosest; each step loosens every limit by 25%."""
        s = min(5, max(1, int(sensitivity)))
        k = 1.25 ** (s - 3)
        d = cls()
        return cls(
            snr=d.snr / k,
            lf_ratio=d.lf_ratio * k,
            hf_share=d.hf_share / k,
            quiet_before=min(0.9, d.quiet_before * k),
            decay_hf=min(0.9, d.decay_hf * k),
            decay_full=min(0.95, d.decay_full * k),
            attack_hops=max(2, round(d.attack_hops * k)),
            crest=d.crest / k,
        )


class SnapAudioDetector:
    """Streaming snap detector. Feed it consecutive mono float32 blocks of any size.

    ``process(block, t_start)`` returns the candidates judged during that block
    (each ~150 ms after its onset). ``nf`` is the HF noise floor (mean square)
    and ``self_gate_until`` rejects onsets before that perf_counter time as
    "own sound" (set it while ZAP plays a sound effect).
    """

    def __init__(self, rate: int = 48000, sensitivity: int = 3) -> None:
        self.rate = int(rate)
        self.hop = max(16, round(128 * self.rate / 48000))
        self.hop_s = self.hop / self.rate
        hi = min(8000.0, 0.45 * self.rate)
        self._sos_h = butter(4, [1500.0, hi], btype="bandpass", fs=self.rate, output="sos")
        self._sos_l = butter(2, 600.0, btype="lowpass", fs=self.rate, output="sos")
        self._ring = 1 << math.ceil(math.log2(2.0 / self.hop_s))  # >= 2 s of hops
        self._init_hops = self._hops(1000)
        self._judge_hops = self._hops(150)
        self._refractory_hops = self._hops(150)
        self._before_hops = (self._hops(200), self._hops(8))
        self._decay_hops = (self._hops(30), self._hops(150))
        self._ten_ms = self._hops(10)
        self._attack_span = self._hops(20)
        self.sensitivity = sensitivity
        self.self_gate_until = -math.inf
        self.reset()

    def _hops(self, ms: float) -> int:
        return max(1, round(ms / 1000.0 / self.hop_s))

    def reset(self) -> None:
        r = self._ring
        self._zi_h = np.zeros((self._sos_h.shape[0], 2))
        self._zi_l = np.zeros((self._sos_l.shape[0], 2))
        self._rest = np.zeros((3, 0))  # unhopped tail of (raw, hf, lf)
        self._eh = np.zeros(r)
        self._el = np.zeros(r)
        self._ef = np.zeros(r)
        self._ph = np.zeros(r)  # peak |hf| per hop, for the crest factor
        self._t = np.zeros(r)  # hop start time
        self._n = 0  # hops written so far
        self.nf = MIN_NF
        self.ready = False  # noise floor initialised
        self._pending: int | None = None
        self._last_trigger = -(10**9)
        self._last_accepted_t = -math.inf

    # ------------------------------------------------------------ settings
    @property
    def sensitivity(self) -> int:
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value: int) -> None:
        self._sensitivity = min(5, max(1, int(value)))
        self.th = Thresholds.for_sensitivity(self._sensitivity)

    def gate(self, until: float) -> None:
        """Reject onsets before ``until`` (perf_counter) as our own sound."""
        self.self_gate_until = max(self.self_gate_until, until)

    @property
    def pending_onset(self) -> float | None:
        """Onset time of the candidate still waiting to be judged, if any."""
        p = self._pending
        return None if p is None else float(self._t[p % self._ring])

    @property
    def noise_floor_dbfs(self) -> float:
        return 10.0 * math.log10(max(self.nf, MIN_NF))

    # ------------------------------------------------------------ streaming
    def process(self, block: np.ndarray, t_start: float) -> list[AudioCandidate]:
        x = np.asarray(block, dtype=np.float64).ravel()
        if x.size == 0:
            return []
        h, self._zi_h = sosfilt(self._sos_h, x, zi=self._zi_h)
        lo, self._zi_l = sosfilt(self._sos_l, x, zi=self._zi_l)
        t0 = t_start - self._rest.shape[1] / self.rate
        sig = np.stack((x, h, lo))
        if self._rest.shape[1]:
            sig = np.concatenate((self._rest, sig), axis=1)
        n = sig.shape[1] // self.hop
        used = n * self.hop
        self._rest = sig[:, used:].copy()
        if n == 0:
            return []
        hops = sig[:, :used].reshape(3, n, self.hop)
        energy = np.einsum("cnk,cnk->cn", hops, hops) / self.hop
        slots = (self._n + np.arange(n)) % self._ring
        self._ef[slots], self._eh[slots], self._el[slots] = energy
        self._ph[slots] = np.abs(hops[1]).max(axis=1)
        self._t[slots] = t0 + np.arange(n) * self.hop_s
        out = []
        for _ in range(n):
            self._n += 1
            c = self._step(self._n - 1)
            if c is not None:
                out.append(c)
        return out

    def _step(self, i: int) -> AudioCandidate | None:
        eh, r = self._eh, self._ring
        e = float(eh[i % r])
        if i < self._init_hops:
            if i == self._init_hops - 1:
                self.nf = max(MIN_NF, float(np.median(self._eh[: self._init_hops])))
                self.ready = True
            return None
        out = None
        if self._pending is not None:
            if i - self._pending < self._judge_hops:
                return None  # noise floor frozen while a candidate is pending
            out = self._judge(self._pending)
            self._pending = None
        nf = self.nf
        prev = max(eh[(i - 1) % r], eh[(i - 2) % r], eh[(i - 3) % r])
        if e > 3.0 * nf and e > MIN_TRIGGER and e > RISE * prev and i - self._last_trigger >= self._refractory_hops:
            self._pending = self._last_trigger = i
        else:
            a = 0.01 if e < 4.0 * nf else 0.0005
            self.nf = max(MIN_NF, nf + a * (e - nf))
        return out

    def _take(self, arr: np.ndarray, a: int, b: int) -> np.ndarray:
        """Ring slice for hop indices [a, b)."""
        return np.take(arr, np.arange(a, b) % self._ring)

    # ------------------------------------------------------------ judging
    def features(self, i0: int) -> tuple[float, dict[str, float]]:
        """Onset time and shape features of the candidate that opened at hop ``i0``."""
        eh, ef, nf = self._eh, self._ef, self.nf
        first = self._take(eh, i0, i0 + 3)
        ip = i0 + int(np.argmax(first))
        peak = max(float(first.max()), TINY)
        peak_f = max(float(self._take(ef, i0, i0 + 3).max()), TINY)

        lead = self._take(eh, i0 - 3, ip + 1)
        start = i0 - 3 + int(np.argmax(lead > peak / 3.0))
        t_onset = float(self._t[start % self._ring])

        span = self._take(eh, i0 - 3, i0 + self._attack_span)
        jp = int(np.argmax(span))
        attack = jp - int(np.argmax(span[: jp + 1] > span[jp] / 3.0))

        lf = float(self._take(self._el, i0, i0 + self._ten_ms).sum())
        hf = float(self._take(eh, i0, i0 + self._ten_ms).sum())

        b0, b1 = self._before_hops
        before = self._take(eh, i0 - b0, i0 - b1 + 1)
        tb = self._take(self._t, i0 - b0, i0 - b1 + 1)
        tail = (tb >= self._last_accepted_t - 0.01) & (tb <= self._last_accepted_t + 0.15)
        before = before[~tail]
        quiet_before = float(before.max()) / peak if before.size else 0.0

        d0, d1 = self._decay_hops
        decay_hf = float(self._take(eh, i0 + d0, i0 + d1 + 1).max()) / peak
        decay_full = float(self._take(ef, i0 + d0, i0 + d1 + 1).max()) / peak_f

        half = self._ten_ms // 2
        rms = math.sqrt(max(float(self._take(eh, ip - half, ip + half).mean()), TINY))
        crest = float(self._take(self._ph, ip - half, ip + half).max()) / rms

        return t_onset, {
            "snr": peak / nf,
            "lf_ratio": lf / max(hf, TINY),
            "hf_share": peak / max(float(ef[ip % self._ring]), TINY),
            "quiet_before": quiet_before,
            "decay_hf": decay_hf,
            "decay_full": decay_full,
            "attack": float(attack),
            "crest": crest,
            "peak_db": 10.0 * math.log10(peak),
            "nf_db": 10.0 * math.log10(nf),
        }

    def _verdict(self, f: dict[str, float]) -> str:
        th = self.th
        if f["snr"] < th.snr:
            return TOO_QUIET
        if f["lf_ratio"] > th.lf_ratio:
            return LF_HEAVY
        if f["hf_share"] < th.hf_share:
            return NOT_HF
        if f["quiet_before"] >= th.quiet_before:
            return NOISY_BEFORE
        if f["decay_hf"] > th.decay_hf:
            return SLOW_DECAY
        if f["decay_full"] > th.decay_full:
            return RINGING
        if f["attack"] > th.attack_hops:
            return SLOW_ATTACK
        if f["crest"] < th.crest:
            return NOT_PEAKY
        return OK

    def _judge(self, i0: int) -> AudioCandidate:
        t_onset, f = self.features(i0)
        reason = self._verdict(f)
        if t_onset < self.self_gate_until:
            reason = OWN_SOUND
        accepted = reason == OK
        if accepted:
            self._last_accepted_t = t_onset
        log.debug("snap candidate %.3f %s %s", t_onset, reason, {k: round(v, 2) for k, v in f.items()})
        return AudioCandidate(t_onset, f, accepted, reason)
