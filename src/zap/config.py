"""User configuration: ``config.toml`` in the ZAP home folder.

The defaults below are the source of truth. A user's file only needs the keys
they want to change; anything unknown is rejected with a "did you mean" hint so
typos don't silently do nothing.
"""

from __future__ import annotations

import difflib
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from zap import paths
from zap.gestures.types import ALL_GESTURES

DEFAULT_TOML = """\
# ZAP configuration. Delete a line to fall back to its default.
# Reload while running with Ctrl+Alt+R.

[app]
armed_on_start = false   # start in plain passthrough; Ctrl+Alt+A arms gestures
arm_grace_s = 3.0        # ignore gestures for this long after arming
preview = true           # local preview window with the HUD (never sent to Zoom)
preview_mirror = true    # mirror the local preview only, like a selfie camera
snap_entry = false       # start INVISIBLE (the meeting sees only the room); snap once to appear.
                         # Also arms gestures at start. Capture a clean background first (Ctrl+Alt+B).

[camera]
name = ""                # "" = first physical camera; devices named "OBS*" are always refused
backend = "auto"         # "auto" (DirectShow, then Media Foundation), "dshow" or "msmf"
mirror = true            # see yourself like a mirror; meme text still reads correctly for everyone.
                         # In Zoom, untick "Mirror my video" so Zoom doesn't flip it back.
width = 1280
height = 720
fps = 30

[output]
width = 1280
height = 720
fps = 30

[perception]
infer_width = 640        # must be a multiple of 4
infer_height = 360
face_every = 2           # run the face model every Nth frame when nothing needs it sooner
mask_every = 3           # refresh the person mask every Nth frame for the background plate
hands_idle_every = 2     # while no hand is visible, look for hands only every Nth frame (saves ~12 ms)

[sound]
enabled = true
output = "speakers"      # "speakers" (only you hear) or "meeting" (needs VB-CABLE)
volume = 0.6

[zoom]
end_for_all = false      # true = host ends the meeting for everyone instead of leaving
dry_run = true           # log "WOULD LEAVE" instead of leaving; set false once tested
confirm = "countdown"    # "countdown" (cancellable 3-2-1) or "thumbs_up" (also needs a thumbs-up)
leave_guard_s = 60.0     # ignore the leave trigger for this long after arming
countdown_s = 3.0
palm_cancels = false     # true = an open palm held up also cancels the outro (Esc, Space or a snap always do)

[snap]
sensitivity = 3          # 1 (strict) .. 5 (loose)
burst_gap_ms = 2000      # snaps closer than this count together (2 snaps = To Be Continued)
av_offset_ms = 80        # video lags audio by roughly this much
vision_only = true       # true = snaps are SEEN by the camera (works in noisy rooms);
                         # false = the mic must also hear them
mic = ""                 # part of the microphone name; "" = the Windows default microphone

[cinema]
layout = "left_right"    # "left_right" or "top_bottom"
font = "timesbd.ttf"

[money]
stamp_text = "DEAL CLOSED"
caption = ""             # optional line under the stamp

[brb]
text = "We'll Be Right Back"
seconds = 0.0            # 0 = as long as the music (assets/sfx/brb.*), else this many seconds
hold_when_away = true    # you walked off: keep the card up until you're back in view

# What each gesture does. Values: an effect name, or "cutaway:<name>".
# Every effect has its own hand sign; hold a sign still for about a second, away from your face.
[bindings]
snap1 = "presence"
snap2 = "outro"
snap3 = "none"           # three snaps: off (slow to tell apart)
two_thumbs_up = "money"  # thumbs up = DEAL CLOSED
peace = "outro"          # peace sign = "To Be Continued" (then leave)
point_up = "brb"         # one finger up ("one sec") = We'll Be Right Back
thumbs_down = "cutaway:michael_scott"  # thumbs down = Michael Scott facepalm
rock = "cutaway:jim"     # rock-on hand (index + pinky up) = Jim stares at the camera
hands_up = "cinema"      # both open hands up beside your head = Absolute Cinema
collar = "cutaway:harvey"  # fingers at your collar = Harvey
cover_face = "sad"       # cover your face with both hands, then drop them = crying MJ
forehead = "none"        # hand on forehead: off (it fired by accident); "cutaway:michael_scott" to use it
eyebrows_up = "none"     # eyebrows up: off (it fired while talking); calibrate (Ctrl+Alt+C), then "cutaway:jim"

# Cutaways cut the outgoing video to a meme image or clip from assets/memes.
[cutaways.michael_scott]
asset = "michael_scott_forehead"
caption = "*facepalm*"
seconds = 2.5

[cutaways.jim]
asset = "jim_stare"
caption = "*stares at camera*"
seconds = 2.5

[cutaways.harvey]
asset = "harvey_car"
caption = "*steps out of the car*"
seconds = 2.5
face_swap = "cutout"     # YOUR head on Harvey: "cutout" (meme-style head cutout), "ai" (face-swap model), false

# Per-gesture threshold overrides, e.g.
# [gestures.hands_up]
# t_on = 0.5
# Sections: general, arbiter (settle_s, gap_s, snap_hold_s, pending_max_s, charge_timeout_s),
# hands_up, two_thumbs_up, peace, point_up, thumbs_down, rock, forehead, cover_face, eyebrows_up,
# collar, cancel, thumbs_up.
[gestures]
"""

# Sections whose keys are user-defined rather than fixed.
_FREEFORM = {"bindings", "cutaways", "gestures"}
# What a [bindings] value may name besides "cutaway:<name>"; "none" switches a gesture off.
EFFECT_TARGETS = ("presence", "outro", "money", "cinema", "sad", "brb", "none")


class ConfigError(ValueError):
    pass


@dataclass
class AppCfg:
    armed_on_start: bool = False
    arm_grace_s: float = 3.0
    preview: bool = True
    preview_mirror: bool = True
    snap_entry: bool = False


@dataclass
class CameraCfg:
    name: str = ""
    backend: str = "auto"
    mirror: bool = True
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class OutputCfg:
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class PerceptionCfg:
    infer_width: int = 640
    infer_height: int = 360
    face_every: int = 2
    mask_every: int = 3
    hands_idle_every: int = 2


@dataclass
class SoundCfg:
    enabled: bool = True
    output: str = "speakers"
    volume: float = 0.6


@dataclass
class ZoomCfg:
    end_for_all: bool = False
    dry_run: bool = True
    confirm: str = "countdown"
    leave_guard_s: float = 60.0
    countdown_s: float = 3.0
    palm_cancels: bool = False


@dataclass
class SnapCfg:
    sensitivity: int = 3
    burst_gap_ms: int = 2000
    av_offset_ms: int = 80
    vision_only: bool = True
    mic: str = ""


@dataclass
class MoneyCfg:
    stamp_text: str = "DEAL CLOSED"
    caption: str = ""


@dataclass
class CinemaCfg:
    layout: str = "left_right"
    font: str = "timesbd.ttf"


@dataclass
class BrbCfg:
    text: str = "We'll Be Right Back"
    seconds: float = 0.0
    hold_when_away: bool = True


@dataclass
class Config:
    app: AppCfg = field(default_factory=AppCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    output: OutputCfg = field(default_factory=OutputCfg)
    perception: PerceptionCfg = field(default_factory=PerceptionCfg)
    sound: SoundCfg = field(default_factory=SoundCfg)
    zoom: ZoomCfg = field(default_factory=ZoomCfg)
    snap: SnapCfg = field(default_factory=SnapCfg)
    cinema: CinemaCfg = field(default_factory=CinemaCfg)
    money: MoneyCfg = field(default_factory=MoneyCfg)
    brb: BrbCfg = field(default_factory=BrbCfg)
    bindings: dict[str, str] = field(default_factory=dict)
    cutaways: dict[str, dict[str, Any]] = field(default_factory=dict)
    gestures: dict[str, dict[str, Any]] = field(default_factory=dict)
    source: Path | None = None


def _suggest(key: str, options) -> str:
    match = difflib.get_close_matches(key, list(options), n=1)
    return f" (did you mean '{match[0]}'?)" if match else ""


def _merge(base: dict, over: dict, where: str) -> dict:
    out = dict(base)
    for key, value in over.items():
        if where in _FREEFORM or (where == "" and key in _FREEFORM):
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                merged = dict(out[key])
                for k, v in value.items():
                    merged[k] = {**merged[k], **v} if isinstance(v, dict) and isinstance(merged.get(k), dict) else v
                out[key] = merged
            else:
                out[key] = value
            continue
        if key not in base:
            path = f"{where}.{key}" if where else key
            raise ConfigError(f"unknown setting '{path}'{_suggest(key, base)}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ConfigError(f"'{key}' must be a [section]")
            out[key] = _merge(base[key], value, key)
        else:
            expected = type(base[key])
            if expected is float and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
                path = f"{where}.{key}" if where else key
                raise ConfigError(f"'{path}' should be {expected.__name__}, got {value!r}")
            out[key] = value
    return out


def _build(data: dict, source: Path | None) -> Config:
    cfg = Config(source=source)
    for f in fields(Config):
        if f.name == "source" or f.name not in data:
            continue
        current = getattr(cfg, f.name)
        if isinstance(current, dict):
            setattr(cfg, f.name, dict(data[f.name]))
        else:
            setattr(cfg, f.name, type(current)(**data[f.name]))
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.perception.infer_width % 4:
        raise ConfigError("perception.infer_width must be a multiple of 4")
    if cfg.output.width % 2 or cfg.output.height % 2:
        raise ConfigError("output width and height must be even")
    if cfg.sound.output not in ("speakers", "meeting"):
        raise ConfigError(f"sound.output must be 'speakers' or 'meeting'{_suggest(cfg.sound.output, ['speakers', 'meeting'])}")
    if cfg.zoom.confirm not in ("countdown", "thumbs_up"):
        raise ConfigError("zoom.confirm must be 'countdown' or 'thumbs_up'")
    if cfg.camera.backend not in ("auto", "dshow", "msmf"):
        raise ConfigError(f"camera.backend must be 'auto', 'dshow' or 'msmf'{_suggest(cfg.camera.backend, ['auto', 'dshow', 'msmf'])}")
    for key in ("face_every", "mask_every", "hands_idle_every"):
        if getattr(cfg.perception, key) < 1:
            raise ConfigError(f"perception.{key} must be at least 1")
    if cfg.cinema.layout not in ("left_right", "top_bottom"):
        raise ConfigError("cinema.layout must be 'left_right' or 'top_bottom'")
    if not 1 <= cfg.snap.sensitivity <= 5:
        raise ConfigError("snap.sensitivity must be between 1 and 5")
    for gesture, target in cfg.bindings.items():
        if gesture not in ALL_GESTURES:
            raise ConfigError(f"[bindings] has unknown gesture '{gesture}'{_suggest(gesture, ALL_GESTURES)}")
        if not isinstance(target, str):
            raise ConfigError(f"binding '{gesture}' should be a string, got {target!r}")
        if target.startswith("cutaway:"):
            name = target.split(":", 1)[1]
            if name not in cfg.cutaways:
                raise ConfigError(f"binding '{gesture}' uses unknown cutaway '{name}'{_suggest(name, cfg.cutaways)}")
        elif target not in EFFECT_TARGETS:
            raise ConfigError(
                f"binding '{gesture}' = '{target}' is not an effect{_suggest(target, EFFECT_TARGETS)}; "
                f"use one of {', '.join(EFFECT_TARGETS)} or cutaway:<name>"
            )


def defaults() -> Config:
    return _build(tomllib.loads(DEFAULT_TOML), None)


def load(path: Path | None = None, create: bool = True) -> Config:
    """Load the user's config merged over the defaults.

    Writes a commented default file on first run so users have something to edit.
    """
    path = path or paths.config_path()
    base = tomllib.loads(DEFAULT_TOML)
    if not path.exists():
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(DEFAULT_TOML, encoding="utf-8")
        return _build(base, path)
    try:
        user = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path.name}: {e}") from e
    return _build(_merge(base, user, ""), path)
