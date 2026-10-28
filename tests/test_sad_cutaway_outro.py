"""Sad mode, cutaways and the outro, run end to end on synthetic frames (no camera, mic or Zoom).

Each effect is driven through its whole timeline at 30 fps of simulated time
with an AssetResolver pointed at a temp folder: once with no assets (procedural
fallbacks) and once with generated meme files. Representative frames are saved
to the system temp folder for a visual check.
"""

from __future__ import annotations

import itertools
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pytest

from zap.assets import AssetResolver
from zap.effects import cutaway as cutaway_mod
from zap.effects import registry
from zap.effects import sad as sad_mod
from zap.effects.base import Effect, FxContext, Layer
from zap.effects.compositor import Compositor
from zap.effects.cutaway import Cutaway
from zap.effects.outro import SEPIA_BGR, Outro
from zap.effects.sad import ANCHORS, SadMode
from zap.fetch_models import model_path
from zap.render.blend import vignette_map
from zap.signals import FACE_LOWER_LIDS, Face, Hand, Signals

W, H = 1280, 720
FPS = 30.0
T0 = 1000.0
P95_LIMIT_MS = 40.0
REAL_FACE_MODEL = model_path("face_landmarker.task")  # resolved before ZAP_HOME is redirected
SNAP_DIR = Path(tempfile.gettempdir())


# ================================================================ synthetic world
def synth_frame(shade: int = 0) -> np.ndarray:
    """A room-ish gradient with a skin-tone face blob and a shirt; `shade` shifts brightness."""
    x = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    f = np.empty((H, W, 3), np.uint8)
    f[..., 0] = (60 + 90 * x + 0 * y).astype(np.uint8)
    f[..., 1] = (80 + 60 * y + 0 * x).astype(np.uint8)
    f[..., 2] = (120 + 70 * (1 - x) * y).astype(np.uint8)
    cv2.ellipse(f, (640, 340), (135, 180), 0, 0, 360, (120, 150, 205), -1, cv2.LINE_AA)
    cv2.ellipse(f, (640, 760), (300, 200), 0, 0, 360, (90, 60, 40), -1, cv2.LINE_AA)
    if shade:
        cv2.add(f, (shade, shade, shade, 0), dst=f)
    return f


# canonical face points in face widths: u right, v down, origin at the nose bridge
_NAMED_UV = {
    234: (-0.5, 0.1), 454: (0.5, 0.1), 168: (0.0, 0.0), 1: (0.0, 0.3),
    33: (-0.32, 0.02), 133: (-0.1, 0.02), 362: (0.1, 0.02), 263: (0.32, 0.02),
    145: (-0.21, 0.07), 374: (0.21, 0.07), 61: (-0.18, 0.52), 291: (0.18, 0.52),
    152: (0.0, 0.82), 10: (0.0, -0.55), 105: (-0.2, -0.12), 334: (0.2, -0.12),
}


def canonical_uv(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(0, 1, 478))
    th = rng.uniform(0, 2 * math.pi, 478)
    uv = np.stack([0.48 * r * np.cos(th), 0.15 + 0.66 * r * np.sin(th)], axis=-1)
    for i, p in _NAMED_UV.items():
        uv[i] = p
    return uv


def synth_face(cx: float = 640, cy: float = 300, fw: float = 270, angle_deg: float = 0.0) -> Face:
    a = math.radians(angle_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    xy = canonical_uv() @ R.T * fw + (cx, cy)
    lm = np.zeros((478, 3), np.float32)
    lm[:, :2] = xy
    return Face(lm=lm, t=0.0)


def synth_hand(wx: float = 560, wy: float = 560) -> Hand:
    """An open hand pointing up, wrist at (wx, wy): five fingers of four joints each."""
    pts = [(wx, wy)]
    for k, dx in enumerate((-55, -25, 0, 25, 50)):
        base_y = wy - (40 if k == 0 else 90)
        for j in range(4):
            pts.append((wx + dx * (1 + 0.25 * j), base_y - 28 * j))
    lm = np.zeros((21, 3), np.float32)
    lm[:, :2] = pts
    return Hand(lm=lm, score=0.95)


class RecordingSfx:
    def __init__(self) -> None:
        self.played: list[str] = []
        self.stops = 0

    def play(self, key: str) -> None:
        self.played.append(key)

    def stop_all(self) -> None:
        self.stops += 1


@dataclass
class Scene:
    assets: AssetResolver
    face: Face | None = None
    hands: list[Hand] = field(default_factory=list)
    varying: bool = False  # the live frame changes every frame (to prove freezes are frozen)
    sfx: RecordingSfx = field(default_factory=RecordingSfx)
    base: np.ndarray = field(default_factory=synth_frame)
    seq: int = 0

    def ctx(self, now: float) -> FxContext:
        self.seq += 1
        frame = synth_frame(shade=(self.seq * 7) % 60) if self.varying else self.base
        sig = Signals(t=now, seq=self.seq, hands=list(self.hands), face=self.face, width=W, height=H)
        return FxContext(frame=frame, sig=sig, now=now, assets=self.assets, sfx=self.sfx)


@dataclass
class Run:
    times_ms: list[float]
    end_t: float | None  # simulated seconds at which the effect went inactive
    snaps: dict[float, np.ndarray]
    frames: int

    @property
    def p95(self) -> float:
        return float(np.percentile(self.times_ms, 95)) if self.times_ms else 0.0


def run(fx: Effect, scene: Scene, t_max: float, snaps: tuple[float, ...] = (), hook=None,
        started: bool = False) -> Run:
    """Start (unless `started`) and drive `fx` at 30 fps until it goes inactive or t_max passes."""
    if not started:
        fx.start(scene.ctx(T0))
    times, got = [], {}
    pending = sorted(snaps)
    n = 0
    while True:
        t = n / FPS
        if t > t_max:
            return Run(times, None, got, n)
        ctx = scene.ctx(T0 + t)
        if hook is not None:
            hook(t, ctx, fx)
        if not fx.active:
            return Run(times, t, got, n)
        out = ctx.frame.copy()
        t1 = time.perf_counter()
        fx.update(ctx, out)
        times.append((time.perf_counter() - t1) * 1000)
        assert out.shape == (H, W, 3) and out.dtype == np.uint8 and out.flags.c_contiguous
        while pending and t >= pending[0]:
            got[pending.pop(0)] = out.copy()
        n += 1


def ends_at(r: Run, t_end: float) -> bool:
    """The effect went inactive at t_end (noticed by the loop one frame later, give or take rounding)."""
    return r.end_t is not None and t_end - 1e-6 <= r.end_t <= t_end + 2 / FPS + 1e-6


def save(name: str, img: np.ndarray) -> Path:
    p = SNAP_DIR / f"zap_fx_{name}.png"
    ok, buf = cv2.imencode(".png", img)
    assert ok
    buf.tofile(str(p))
    print(f"saved {p}")
    return p


def report(name: str, r: Run) -> None:
    print(f"{name}: {r.frames} frames, p95 {r.p95:.2f} ms, max {max(r.times_ms, default=0):.2f} ms")
    assert r.p95 < P95_LIMIT_MS, f"{name} p95 {r.p95:.1f} ms"


# ================================================================ fixtures
@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """Isolate ZAP_HOME (calibration cache) and the sad module's process-wide meme caches."""
    h = tmp_path / "zap_home"
    monkeypatch.setenv("ZAP_HOME", str(h))
    monkeypatch.setattr(sad_mod, "_MEMES", {})
    monkeypatch.setattr(sad_mod, "_BY_ID", {})
    return h


@pytest.fixture
def empty_assets(tmp_path, home) -> AssetResolver:
    memes, sfx = tmp_path / "memes", tmp_path / "sfx"
    memes.mkdir()
    sfx.mkdir()
    return AssetResolver(memes, sfx)


@pytest.fixture
def assets(empty_assets) -> AssetResolver:
    """Generated meme files: a cutout PNG, a still JPG and a short clip."""
    memes = empty_assets.memes
    write_img(memes / "crying_jordan.png", synth_cutout())
    write_img(memes / "jim_stare.jpg", synth_still())
    write_clip(memes / "harvey_button", seconds=1.5, fps=20)
    write_clip(memes / "loop_gif", seconds=0.5, fps=10)
    return empty_assets


def write_img(p: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(p.suffix, img)
    assert ok
    buf.tofile(str(p))


def synth_cutout() -> np.ndarray:
    """A cartoon crying face on a transparent background (BGRA)."""
    img = np.zeros((480, 400, 4), np.uint8)
    cv2.ellipse(img, (200, 250), (150, 200), 0, 0, 360, (70, 105, 160, 255), -1, cv2.LINE_AA)
    for ex in (140, 260):
        cv2.ellipse(img, (ex, 210), (34, 16), 0, 0, 360, (30, 30, 40, 255), -1, cv2.LINE_AA)
    cv2.ellipse(img, (200, 360), (70, 38), 0, 0, 360, (40, 30, 90, 255), -1, cv2.LINE_AA)
    cv2.line(img, (200, 230), (200, 300), (50, 80, 130, 255), 6, cv2.LINE_AA)
    return img


def synth_still() -> np.ndarray:
    img = np.zeros((600, 800, 3), np.uint8)
    img[:] = (40, 120, 200)
    cv2.rectangle(img, (250, 150), (550, 450), (230, 230, 230), -1)
    cv2.putText(img, "JIM", (300, 330), cv2.FONT_HERSHEY_SIMPLEX, 3, (20, 20, 20), 6)
    return img


def write_clip(stem: Path, seconds: float, fps: int) -> Path:
    """A tiny clip whose frames get brighter; mp4 (mp4v) if this OpenCV can write it, else MJPG avi."""
    n = int(seconds * fps)
    for ext, fourcc in ((".mp4", "mp4v"), (".avi", "MJPG")):
        p = stem.with_suffix(ext)
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*fourcc), fps, (320, 240))
        if not vw.isOpened():
            continue
        for i in range(n):
            f = np.full((240, 320, 3), 20 + int(200 * i / max(1, n - 1)), np.uint8)
            cv2.circle(f, (160, 120), 40, (0, 0, 255), -1)
            vw.write(f)
        vw.release()
        if p.is_file() and p.stat().st_size > 0:
            return p
        p.unlink(missing_ok=True)
    pytest.skip("this OpenCV build cannot write a test clip")


# ================================================================ cutaway
def test_cutaway_card_fallback_runs_for_seconds(empty_assets):
    scene = Scene(empty_assets)
    fx = Cutaway(name="jim", asset="jim_stare", caption="*stares at camera*", seconds=2.5)
    r = run(fx, scene, 6.0, snaps=(0.0, 1.2))
    assert fx.mode == "card"
    assert scene.sfx.played == ["cutaway_jim"]
    assert ends_at(r, fx.t_in + 2.5 + fx.t_out)
    held = r.snaps[1.2]
    assert held.mean() < scene.base.mean() * 0.7  # dark card, not the live frame
    assert cv2.absdiff(held, scene.base).mean() > 20
    assert cv2.absdiff(r.snaps[0.0], scene.base).mean() < 3  # fades in from what the meeting saw
    report("cutaway card", r)
    save("cutaway_card", held)


def test_cutaway_image_morphs_in_holds_and_morphs_out(assets):
    scene = Scene(assets)
    fx = Cutaway(name="jim", asset="jim_stare", seconds=2.5)
    total = fx.t_in + 2.5 + fx.t_out
    r = run(fx, scene, 6.0, snaps=(0.0, 1.5, total - 0.05))
    assert fx.mode == "image"
    assert ends_at(r, total)
    shown = cutaway_mod.present(synth_still(), W, H)
    assert cv2.absdiff(r.snaps[0.0], scene.base).mean() < 3  # starts from the live picture
    assert cv2.absdiff(r.snaps[1.5], shown).mean() < 6  # the whole meme while held, never cropped
    back = cv2.absdiff(r.snaps[total - 0.05], scene.base).mean()
    assert back < cv2.absdiff(r.snaps[1.5], scene.base).mean()  # morphing back to the live picture
    assert ("jim_stare", W, H) in cutaway_mod._FITTED[assets]
    report("cutaway image", r)
    save("cutaway_image", r.snaps[1.5])


def test_present_fits_whole_meme_over_blurred_backdrop():
    square = np.zeros((300, 300, 3), np.uint8)
    square[:] = (40, 160, 220)
    out = cutaway_mod.present(square, W, H)
    assert out.shape == (H, W, 3)
    assert np.abs(out[H // 2, W // 2].astype(int) - (40, 160, 220)).max() <= 3  # the meme, centred
    assert out[H // 2, 10].mean() < square[0, 0].mean()  # side bands: darkened backdrop, not black bars


def test_cutaway_clip_plays_at_clip_fps(assets):
    scene = Scene(assets)
    fx = Cutaway(name="harvey", asset="harvey_button", seconds=2.5)
    r = run(fx, scene, 6.0, snaps=(0.6, 1.2))
    assert fx.mode == "clip"
    assert ends_at(r, fx.duration)
    assert r.snaps[1.2].mean() > r.snaps[0.6].mean() + 30  # later clip frames are brighter
    report("cutaway clip", r)
    save("cutaway_clip", r.snaps[1.2])


def test_cutaway_short_clip_loops_until_seconds(assets):
    fx = Cutaway(name="loop", asset="loop_gif", seconds=2.0)
    r = run(fx, Scene(assets), 6.0)
    assert fx.mode == "clip"
    assert ends_at(r, fx.t_in + 2.0 + fx.t_out)


def test_cutaway_from_registry_with_config_section(empty_assets):
    registry.load_builtin()
    section = {"asset": "michael_scott_forehead", "caption": "*pinches forehead*", "seconds": 1.0}
    fx = registry.create("cutaway", cutaway="michael_scott", **section)
    assert isinstance(fx, Cutaway) and fx.name == "cutaway" and fx.layer == Layer.CUTAWAY
    scene = Scene(empty_assets)
    r = run(fx, scene, 3.0)
    assert scene.sfx.played == ["cutaway_michael_scott"]
    assert ends_at(r, fx.t_in + 1.0 + fx.t_out)
    assert "assets/memes/michael_scott_forehead.png" in empty_assets.missing


# ================================================================ sad
def test_sad_fallback_full_timeline(empty_assets):
    face = synth_face()
    scene = Scene(empty_assets, face=face)
    fx = SadMode(seed=1)
    tear_counts = []
    r = run(fx, scene, 8.0, snaps=(3.0, 6.2), hook=lambda t, ctx, f: tear_counts.append(f._tears.count))
    assert fx.meme is None
    assert scene.sfx.played == ["sad"]
    assert ends_at(r, 6.4)
    assert max(tear_counts) >= 4
    mid = r.snaps[3.0]
    b, rch = float(mid[..., 0].mean()), float(mid[..., 2].mean())
    sb, sr = float(scene.base[..., 0].mean()), float(scene.base[..., 2].mean())
    assert (b - rch) > (sb - sr)  # cooler than the input
    assert mid.mean() < scene.base.mean()  # darker
    # fading out: closer to the input than mid-effect
    assert cv2.absdiff(r.snaps[6.2], scene.base).mean() < cv2.absdiff(mid, scene.base).mean()
    report("sad fallback", r)
    save("sad_fallback", mid)


def test_sad_tears_spawn_at_lower_lids_and_fall(empty_assets):
    face = synth_face()
    fx = SadMode(seed=2)
    run(fx, Scene(empty_assets, face=face), 1.0)
    tears = fx._tears
    alive = np.flatnonzero(tears.alive)
    assert len(alive) > 0
    lids_y = face.lm[list(FACE_LOWER_LIDS), 1].mean()
    assert (tears.vel[alive, 1] > 0).all()  # falling
    assert tears.pos[alive, 1].min() >= lids_y - 40  # never above the eyes (zoom shifts slightly)


def test_sad_without_face_still_grades(empty_assets):
    fx = SadMode(seed=3)
    r = run(fx, Scene(empty_assets, face=None), 8.0)
    assert r.end_t is not None and fx._tears.count == 0


def test_sad_meme_fallback_alignment_and_hand_occlusion(assets, monkeypatch, tmp_path):
    """No face model -> the cutout is aligned by its alpha box; the hand stays in front of it."""
    monkeypatch.setattr(sad_mod, "model_path", lambda name: tmp_path / "missing" / name)
    face, hand = synth_face(), synth_hand(wx=540, wy=600)
    with_meme = Scene(assets, face=face, hands=[hand])
    fx = SadMode(seed=5)
    r = run(fx, with_meme, 8.0, snaps=(0.5, 3.0))
    assert fx.meme is not None and fx.meme.ready.wait(5) and fx.meme.lm is None
    assert r.end_t is not None

    no_meme_assets = AssetResolver(tmp_path / "nothing", tmp_path / "nothing")
    ref = run(SadMode(seed=5), Scene(no_meme_assets, face=face, hands=[hand]), 0.5, snaps=(0.5,))
    a, b = r.snaps[0.5], ref.snaps[0.5]
    nose = face.lm[1, :2].astype(int)
    palm = hand.palm_center.astype(int)
    patch = lambda img, p: img[p[1] - 6:p[1] + 6, p[0] - 6:p[0] + 6].astype(np.int16)
    assert np.abs(patch(a, nose) - patch(b, nose)).mean() > 10  # meme drawn on the face
    assert np.abs(patch(a, palm) - patch(b, palm)).mean() < 2  # but not over the hand
    report("sad meme (bbox fallback)", r)
    save("sad_meme", r.snaps[3.0])


@pytest.mark.skipif(not REAL_FACE_MODEL.is_file(), reason="face_landmarker.task not downloaded")
def test_sad_meme_landmarks_detected_once_and_cached(assets, home, monkeypatch):
    """The real FaceLandmarker runs in IMAGE mode off-thread; the result (even 'no face') is cached."""
    monkeypatch.setattr(sad_mod, "model_path", lambda name: REAL_FACE_MODEL)
    scene = Scene(assets, face=synth_face())
    fx = SadMode(seed=7)
    r = run(fx, scene, 1.0)  # renders on the fallback while the model loads
    mf = fx.meme
    assert mf is not None and mf.ready.wait(60)
    cached = list((home / "calibration").glob("meme_landmarks_*.json"))
    assert len(cached) == 1 and cached[0].name == f"meme_landmarks_{mf.sha[:12]}.json"
    assert mf.lm is None or mf.lm.shape == (478, 2)  # a cartoon may or may not count as a face
    # a second effect reuses the prepared meme without another detection thread
    fx2 = SadMode(seed=8)
    r2 = run(fx2, scene, 8.0)
    assert fx2.meme is mf and r2.end_t is not None
    report("sad meme (after detection)", r2)
    assert r.frames > 0


def test_sad_similarity_recovers_known_transform():
    """With meme landmarks, RANSAC on the anchor points recovers the meme -> live similarity."""
    img = synth_cutout()
    mf, _ = sad_mod._prepare_meme(img, "f" * 40)
    src = canonical_uv() * 260 + (200, 160)
    mf.set_landmarks(src.astype(np.float32), opaque=False)
    s, th, t = 0.8, math.radians(12), np.array([330.0, 140.0])
    R = s * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    live = (src @ R.T + t).astype(np.float32)
    live[ANCHORS[0]] += (25, -30)  # one bad anchor: RANSAC shrugs it off
    M = sad_mod._similarity(mf, live)
    assert M is not None
    assert np.allclose(M[:, :2], R, atol=0.02) and np.allclose(M[:, 2], t, atol=4)
    al = sad_mod._Aligner()
    for i in range(10):
        Ms = al.update(mf, live, T0 + i / FPS)
    assert np.allclose(Ms[:, :2], M[:, :2] * sad_mod.COVER_SCALE, atol=0.01)
    c = np.array([mf.bgra.shape[1] / 2, mf.bgra.shape[0] / 2, 1.0])
    assert np.allclose(Ms @ c, M @ c, atol=1.0)  # same centre, just bigger
    assert al.hold(T0 + 9 / FPS + 0.1) is not None and al.hold(T0 + 5.0) is None


def test_sad_landmark_cache_roundtrip(home):
    lm = np.random.default_rng(0).uniform(0, 300, (478, 2)).astype(np.float32)
    sad_mod._save_cached("a" * 40, lm)
    got = sad_mod._load_cached("a" * 40)
    assert isinstance(got, np.ndarray) and np.allclose(got, lm, atol=0.01)
    sad_mod._save_cached("b" * 40, None)
    assert sad_mod._load_cached("b" * 40) is None  # "no face" is remembered too
    assert sad_mod._load_cached("c" * 40) is sad_mod._MISS


def test_cut_hand_clears_alpha_under_hand():
    alpha = np.full((400, 400), 255, np.uint8)
    hand = synth_hand(wx=200, wy=330)
    sad_mod._cut_hand(alpha, hand.lm[:, :2])
    pc = hand.palm_center.astype(int)
    assert alpha[pc[1], pc[0]] == 0 and alpha[5, 5] == 255


def test_tear_stamp_handles_edges_and_views():
    out = np.zeros((50, 60, 3), np.uint8)
    pm, a, oy, ox, _, _ = sad_mod._tear_sprite(12)
    xs = np.int32([-3, 20, 55])
    ys = np.int32([-5, 20, 45])
    sad_mod.stamp(out, pm, a, oy, ox, xs, ys, np.ones(3, np.float32))
    assert out[20:32, 20:28].max() > 100 and out[0:2, 0:2].max() > 0
    view = np.zeros((50, 120, 3), np.uint8)[:, ::2]  # non-contiguous destination
    sad_mod.stamp(view, pm, a, oy, ox, xs, ys, np.ones(3, np.float32))
    assert view[20:32, 20:28].max() > 100


# ================================================================ outro
def test_outro_countdown_fires_once_and_holds_freeze(empty_assets):
    calls: list[float] = []
    scene = Scene(empty_assets, varying=True)
    states: list[tuple[float, str, float]] = []
    fx = Outro(countdown_s=3.0, on_finish=lambda: calls.append(fx_now[0]))
    fx_now = [0.0]

    def hook(t, ctx, f):
        fx_now[0] = t
        states.append((t, f.state, f.seconds_left))

    r = run(fx, scene, 15.0, snaps=(0.0, 1.0, 1.5, 2.0, 3.5), hook=hook)
    assert scene.sfx.played == ["outro"]
    assert len(calls) == 1 and 3.0 <= calls[0] < 3.0 + 1.5 / FPS
    assert ends_at(r, 13.0)  # held 10 s after zero
    assert fx.state == "done" and not fx.active and fx.finished
    counting = [s for s in states if s[1] == "counting"]
    assert counting and all(a[2] >= b[2] for a, b in itertools.pairwise(counting))
    assert any(s[1] == "leaving" and s[2] == 0.0 for s in states)
    # frozen: the live frame keeps changing but the top band (no arrow/numeral) doesn't
    band = lambda img: img[:120, :600]
    assert np.array_equal(band(r.snaps[1.0]), band(r.snaps[2.0]))
    assert np.array_equal(band(r.snaps[1.0]), band(r.snaps[3.5]))
    b, g, rch = (float(band(r.snaps[1.0])[..., i].mean()) for i in range(3))
    assert b < g < rch  # sepia
    assert not np.array_equal(r.snaps[1.0], r.snaps[1.5])  # numerals change
    report("outro countdown", r)
    save("outro_countdown", r.snaps[1.5])


def test_outro_cancel_prevents_on_finish(empty_assets):
    calls = []
    scene = Scene(empty_assets)
    fx = Outro(countdown_s=3.0, on_finish=lambda: calls.append(1))
    results = []

    def hook(t, ctx, f):
        if abs(t - 1.5) < 1e-6:
            results.append(f.cancel(ctx))
            results.append(f.cancel(ctx))  # second cancel is a no-op

    r = run(fx, scene, 6.0, snaps=(1.8,), hook=hook)
    assert results == [True, False]
    assert calls == [] and not fx.finished and fx.state == "cancelled"
    assert scene.sfx.stops == 1
    assert ends_at(r, 1.5 + 0.6)
    save("outro_cancel", r.snaps[1.8])


def test_outro_cancel_without_ctx_from_another_thread(empty_assets):
    calls = []
    fx = Outro(on_finish=lambda: calls.append(1))
    hook = lambda t, ctx, f: f.cancel() if abs(t - 1.0) < 1e-6 else None
    r = run(fx, Scene(empty_assets), 6.0, hook=hook)
    assert calls == [] and r.end_t is not None and r.end_t < 1.0 + 0.6 + 2 / FPS


def test_outro_thumbs_up_timeout_cancels(empty_assets):
    calls = []
    scene = Scene(empty_assets)
    fx = Outro(confirm="thumbs_up", on_finish=lambda: calls.append(1))
    states = []
    r = run(fx, scene, 12.0, snaps=(1.0,), hook=lambda t, ctx, f: states.append(f.state))
    assert calls == [] and not fx.finished
    assert states[0] == "confirming" and "counting" not in states and "cancelled" in states
    assert ends_at(r, 5.0 + 0.6)
    assert scene.sfx.played == []  # no outro sting until confirmed
    assert cv2.absdiff(r.snaps[1.0], scene.base).mean() > 0.5  # the prompt is visible
    save("outro_confirm_prompt", r.snaps[1.0])


def test_outro_thumbs_up_confirm_then_countdown(empty_assets):
    calls = []
    fx = Outro(confirm="thumbs_up", countdown_s=3.0, on_finish=lambda: calls.append(now[0]))
    now = [0.0]

    def hook(t, ctx, f):
        now[0] = t
        if abs(t - 1.0) < 1e-6:
            assert f.confirm() is True  # from the gesture thread: applied next frame
        if abs(t - 2.0) < 1e-6:
            assert f.confirm() is False  # already counting

    scene = Scene(empty_assets)
    run(fx, scene, 6.0, hook=hook)
    assert len(calls) == 1 and 4.0 <= calls[0] < 4.0 + 2 / FPS
    assert fx.state == "leaving" and scene.sfx.played == ["outro"]
    assert fx.cancel() is False and fx.state == "leaving"  # too late to cancel


def test_outro_on_finish_exception_is_contained(empty_assets):
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("zoom window not found")

    fx = Outro(countdown_s=1.0, on_finish=boom)
    r = run(fx, Scene(empty_assets), 3.0)
    assert calls == [1] and fx.state == "leaving" and r.end_t is None


def test_outro_freezes_the_composited_frame():
    """Inside the compositor the freeze includes the lower layers (here a red LOOK)."""

    class RedLook(Effect):
        layer = Layer.LOOK

        def update(self, ctx, out):
            out[:, :, :] = (0, 0, 255)

    comp = Compositor()
    scene = Scene(AssetResolver(Path(tempfile.gettempdir()) / "zap_none", Path(tempfile.gettempdir()) / "zap_none"))
    comp.start(RedLook(), scene.ctx(T0))
    fx = Outro(countdown_s=3.0)
    comp.start(fx, scene.ctx(T0))
    for i in range(20):
        out = comp.render(scene.ctx(T0 + i / FPS))
    red = np.full((H, W, 3), (0, 0, 255), np.uint8)
    expect = cv2.multiply(cv2.transform(red, SEPIA_BGR), vignette_map(W, H, strength=0.45, radius=0.9),
                          scale=1 / 255)
    band = (slice(0, 100), slice(0, 600))  # clear of the arrow and the numerals
    assert cv2.absdiff(out[band], expect[band]).mean() < 3
    assert cv2.absdiff(out[band], scene.base[band]).mean() > 30


def test_prewarm_builds_caches(assets):
    SadMode.prewarm(W, H, assets)
    Outro.prewarm(W, H, countdown_s=3.0)
    Cutaway.prewarm(W, H, assets, {"jim": {"asset": "jim_stare", "caption": "*stares*", "seconds": 2.5},
                                   "harvey": {"asset": "harvey_button"}, "nope": {"asset": "missing"}})
    assert ("jim_stare", W, H) in cutaway_mod._FITTED[assets]
    assert ("clip", "harvey_button") in assets._cache
    fx = Outro(countdown_s=3.0)
    r = run(fx, Scene(assets), 1.0)
    assert max(r.times_ms) < P95_LIMIT_MS  # no cold-start hitch on the freeze frame


def test_outro_rejects_bad_confirm_mode():
    with pytest.raises(ValueError):
        Outro(confirm="maybe")
