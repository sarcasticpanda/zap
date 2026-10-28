"""SfxPlayer and mix_voices with a fake sounddevice: nothing is played or recorded."""

from __future__ import annotations

import time
import types
from typing import ClassVar

import numpy as np
import pytest

from zap.devices import audio_out
from zap.devices.audio_out import GATE_TAIL_S, SAMPLE_RATE, SfxPlayer, Voice, mix_voices

# ---------------------------------------------------------------- mix_voices


def test_mix_single_voice_scaled_and_soft_clipped():
    data = np.linspace(-0.5, 0.5, 100, dtype=np.float32)
    out = np.zeros((64, 2), np.float32)
    left = mix_voices(out, [Voice(data)], 0.5)
    expect = np.tanh(0.5 * data[:64])
    np.testing.assert_allclose(out[:, 0], expect, atol=1e-6)
    np.testing.assert_allclose(out[:, 1], expect, atol=1e-6)
    assert len(left) == 1 and left[0].pos == 64


def test_mix_finishes_voice_and_zero_pads():
    data = np.full(40, 0.25, np.float32)
    out = np.zeros((64, 2), np.float32)
    left = mix_voices(out, [Voice(data, pos=10)], 1.0)
    assert left == []
    np.testing.assert_allclose(out[:30, 0], np.tanh(0.25), atol=1e-6)
    assert not out[30:].any()


def test_mix_sums_voices_with_gain():
    a = np.full(64, 0.1, np.float32)
    b = np.full(64, 0.2, np.float32)
    out = np.zeros((64, 2), np.float32)
    left = mix_voices(out, [Voice(a), Voice(b, gain=0.5)], 1.0)
    np.testing.assert_allclose(out[:, 0], np.tanh(0.2), atol=1e-6)
    assert left == []  # both ended exactly at the block edge


def test_mix_limits_loud_sums_and_keeps_existing_signal():
    loud = np.full(32, 3.0, np.float32)
    out = np.full((32, 2), 0.5, np.float32)  # e.g. microphone already in the block
    mix_voices(out, [Voice(loud), Voice(loud)], 1.0)
    assert np.all(out < 1.0) and np.all(out > 0.99)
    mic = np.full((32, 2), 0.1, np.float32)
    mix_voices(mic, [], 0.6)
    np.testing.assert_allclose(mic, np.tanh(0.1), atol=1e-6)


def test_mix_does_not_mutate_input_voices():
    data = np.ones(10, np.float32)
    v = Voice(data)
    mix_voices(np.zeros((4, 2), np.float32), [v], 1.0)
    assert v.pos == 0 and data[0] == 1.0


# ---------------------------------------------------------------- fake sounddevice


class FakeStream:
    instances: ClassVar[list[FakeStream]] = []
    fail = False

    def __init__(self, samplerate, channels, dtype, device, latency, callback, extra_settings=None):
        if FakeStream.fail:
            raise RuntimeError("PortAudio error")
        self.samplerate, self.channels, self.dtype, self.device = samplerate, channels, dtype, device
        self.callback = callback
        self.latency = 0.02 if not isinstance(channels, tuple) else (0.02, 0.03)
        self.active = False
        self.closed = False
        FakeStream.instances.append(self)

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.closed = True


class FakeDuplex(FakeStream):
    pass


def make_sd(cable: bool) -> types.SimpleNamespace:
    devices = [
        {"name": "Mic (MME)", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0},
        {"name": "Speakers (MME)", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2},
        {"name": "Microphone Array", "hostapi": 1, "max_input_channels": 2, "max_output_channels": 0},
        {"name": "Speakers", "hostapi": 1, "max_input_channels": 0, "max_output_channels": 2},
    ]
    if cable:
        devices.append({"name": "CABLE Input (VB-Audio Virtual Cable)", "hostapi": 1,
                        "max_input_channels": 0, "max_output_channels": 2})
    apis = [
        {"name": "MME", "default_input_device": 0, "default_output_device": 1},
        {"name": "Windows WASAPI", "default_input_device": 2, "default_output_device": 3},
    ]
    return types.SimpleNamespace(
        query_hostapis=lambda i=None: apis if i is None else apis[i],
        query_devices=lambda i=None: devices if i is None else devices[i],
        OutputStream=FakeStream,
        Stream=FakeDuplex,
        WasapiSettings=lambda auto_convert=False: ("wasapi", auto_convert),
    )


class FakeAssets:
    def __init__(self, sounds: dict[str, np.ndarray]) -> None:
        self.sounds = sounds

    def sound(self, key):
        return self.sounds.get(key)


def synth(key: str) -> np.ndarray:
    return np.full(SAMPLE_RATE // 2, 0.1, np.float32)  # 0.5 s


@pytest.fixture
def fake_sd(monkeypatch):
    FakeStream.instances = []
    FakeStream.fail = False

    def use(cable: bool = False):
        monkeypatch.setattr(audio_out, "sd", make_sd(cable))

    use()
    return use


def run_cb(stream: FakeStream, frames: int = 480, mic: float | None = None) -> np.ndarray:
    out = np.zeros((frames, 2), np.float32)
    if mic is None:
        stream.callback(out, frames, None, None)
    else:
        stream.callback(np.full((frames, 2), mic, np.float32), out, frames, None, None)
    return out


def test_speakers_opens_wasapi_default_stereo_48k(fake_sd):
    with SfxPlayer(None, synth) as p:
        s = FakeStream.instances[-1]
        assert p.mode == "speakers" and p.warning == ""
        assert (s.samplerate, s.channels, s.dtype, s.device) == (48000, 2, "float32", 3)
        assert s.active
    assert s.closed


def test_play_prefers_asset_then_synth_and_sets_busy_until(fake_sd):
    asset = np.full(4800, 0.2, np.float32)  # 0.1 s
    with SfxPlayer(FakeAssets({"money": asset}), synth, volume=1.0) as p:
        s = FakeStream.instances[-1]
        assert not p.playing
        t0 = time.perf_counter()
        p.play("money")
        assert p.playing
        assert p.busy_until == pytest.approx(t0 + 0.02 + 0.1 + GATE_TAIL_S, abs=0.05)
        out = run_cb(s)
        np.testing.assert_allclose(out[:, 0], np.tanh(0.2), atol=1e-6)
        p.play("cinema")  # no asset: synth fallback (0.5 s)
        assert p.busy_until == pytest.approx(time.perf_counter() + 0.02 + 0.5 + GATE_TAIL_S, abs=0.05)


def test_charge_does_not_extend_busy_until(fake_sd):
    with SfxPlayer(None, synth) as p:
        p.play("charge")
        assert p.busy_until == 0.0 and not p.playing
        assert np.any(run_cb(FakeStream.instances[-1]))


def test_disabling_stops_voices_immediately(fake_sd):
    with SfxPlayer(None, synth) as p:
        s = FakeStream.instances[-1]
        p.play("money")
        assert np.any(run_cb(s))
        p.set_enabled(False)
        assert not np.any(run_cb(s))
        p.play("money")  # ignored while disabled
        assert not np.any(run_cb(s))
        p.set_enabled(True)
        p.play("money")
        assert np.any(run_cb(s))


def test_disabled_at_start_opens_nothing_until_enabled(fake_sd):
    with SfxPlayer(None, synth, enabled=False) as p:
        assert FakeStream.instances == []
        p.set_enabled(True)
        assert len(FakeStream.instances) == 1 and p.mode == "speakers"


def test_meeting_without_cable_falls_back_to_speakers(fake_sd):
    with SfxPlayer(None, synth, output="meeting") as p:
        assert p.mode == "speakers"
        assert "CABLE" in p.warning
        assert type(FakeStream.instances[-1]) is FakeStream


def test_meeting_with_cable_mixes_mic_and_sfx(fake_sd):
    fake_sd(cable=True)
    with SfxPlayer(None, synth, output="meeting", volume=1.0) as p:
        s = FakeStream.instances[-1]
        assert p.mode == "meeting" and p.warning == ""
        assert isinstance(s, FakeDuplex) and s.device == (2, 4) and s.channels == (2, 2)
        np.testing.assert_allclose(run_cb(s, mic=0.3), np.tanh(0.3), atol=1e-6)  # mic passes through
        p.play("sad")
        np.testing.assert_allclose(run_cb(s, mic=0.3), np.tanh(0.4), atol=1e-6)


def test_meeting_mode_keeps_mic_running_when_muted(fake_sd):
    fake_sd(cable=True)
    with SfxPlayer(None, synth, enabled=False, output="meeting") as p:
        assert p.mode == "meeting"
        p.play("money")
        np.testing.assert_allclose(run_cb(FakeStream.instances[-1], mic=0.2), np.tanh(0.2), atol=1e-6)


def test_device_errors_never_raise(fake_sd):
    FakeStream.fail = True
    with SfxPlayer(None, synth) as p:
        assert p.mode == "none" and p.warning
        p.play("money")  # silently ignored
        p.stop_all()
        p.set_enabled(False)


def test_callback_errors_output_silence(fake_sd):
    def bad_synth(key):
        return np.ones((3, 3), np.float32)  # wrong shape: mixing raises inside the callback

    with SfxPlayer(None, bad_synth) as p:
        p.play("x")
        out = run_cb(FakeStream.instances[-1])
        assert not out.any()


def test_without_sounddevice(monkeypatch):
    monkeypatch.setattr(audio_out, "sd", None)
    p = SfxPlayer(None, synth)
    assert p.mode == "none" and p.warning
    p.play("money")
    p.close()


def test_cutaway_sound_falls_back_to_the_generic_cutaway_sound(fake_sd):
    from zap.devices.audio_out import fallback_keys

    assert fallback_keys("cutaway_jim") == ("cutaway_jim", "cutaway")
    assert fallback_keys("money") == ("money",)
    asked = []

    def synth_spy(key):
        asked.append(key)
        return synth(key)

    generic = np.full(4800, 0.3, np.float32)
    own = np.full(4800, 0.4, np.float32)
    with SfxPlayer(FakeAssets({"cutaway": generic}), synth_spy, volume=1.0) as p:
        assert p._resolve("cutaway_jim") is generic  # no cutaway_jim.wav: the user's cutaway.wav
    with SfxPlayer(FakeAssets({"cutaway": generic, "cutaway_jim": own}), synth_spy) as p:
        assert p._resolve("cutaway_jim") is own
    with SfxPlayer(FakeAssets({}), synth_spy) as p:
        p._resolve("cutaway_harvey")
    assert asked == ["cutaway"]  # the synth's cutaway thud, not the unknown-key blip
