"""Live demo of ZAP's effects on your own webcam, before the full app is wired.

Shows exactly what Zoom would receive (not mirrored). Pose gestures are live;
snaps are on keys here (the full app listens to the microphone).

    keys: 1 snap x1 (Thanos vanish / reappear)   2 snap x3 (money rain)
          3 Absolute Cinema (press again to end)  4 sad / Crying MJ
          5 Michael Scott   6 Jim   7 Harvey      8 snap x2 (To Be Continued)
          0 cancel outro    b capture clean background (step out for 3 s)
          s sound on/off    q quit
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np

from zap.assets import AssetResolver
from zap.config import defaults
from zap.devices.audio_out import SfxPlayer
from zap.effects import registry
from zap.effects.base import FxContext
from zap.effects.compositor import Compositor
from zap.effects.plate import BackgroundPlate
from zap.effects.thanos import Presence
from zap.gestures.arbiter import Arbiter
from zap.gestures.poses import PoseDetector
from zap.render.synth_sfx import synth
from zap.signals import FramePacket
from zap.vision.perception import Perception

W, H = 1280, 720
CUTAWAY_KEYS = {ord("5"): "michael_scott", ord("6"): "jim", ord("7"): "harvey"}


def open_camera() -> cv2.VideoCapture:
    for _ in range(20):
        for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
                cap.set(cv2.CAP_PROP_FPS, 30)
                if cap.read()[0]:
                    return cap
            cap.release()
        print("camera busy, retrying...", flush=True)
        time.sleep(3)
    raise SystemExit("could not open the webcam (close other camera apps)")


def main() -> None:
    cfg = defaults()
    cfg.app.armed_on_start, cfg.app.arm_grace_s = True, 0.0
    registry.load_builtin()
    assets = AssetResolver()
    sfx = SfxPlayer(assets, synth, enabled=True, output="speakers", volume=0.5)
    perception = Perception(cfg.perception, W, H)
    plate = BackgroundPlate(W, H)
    presence = Presence()
    comp = Compositor(presence)
    poses, arbiter = PoseDetector(cfg.gestures), Arbiter(cfg)
    cap = open_camera()
    print(__doc__)

    def cutaway(name: str):
        return registry.create("cutaway", name=name, **cfg.cutaways[name])

    def leave() -> None:
        print(">>> WOULD LEAVE the Zoom meeting now (demo / dry run)", flush=True)

    def fire(gesture: str, ctx: FxContext) -> None:
        target = cfg.bindings.get(gesture, "")
        try:
            if target == "presence":
                presence.toggle(ctx)
            elif target == "outro":
                comp.start(registry.create("outro", countdown_s=cfg.zoom.countdown_s, on_finish=leave), ctx)
            elif target.startswith("cutaway:"):
                comp.start(cutaway(target.split(":", 1)[1]), ctx)
            elif target:
                comp.start(registry.create(target), ctx)
            print(f"[{time.strftime('%H:%M:%S')}] {gesture} -> {target}", flush=True)
        except Exception as e:  # keep the demo alive; report the bug
            print(f"[{time.strftime('%H:%M:%S')}] {gesture} -> {target} FAILED: {type(e).__name__}: {e}", flush=True)

    cv2.namedWindow("ZAP live demo (what Zoom sees)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ZAP live demo (what Zoom sees)", 1120, 630)
    seq, t_prev, fps, sound = 0, time.perf_counter(), 0.0, True
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if frame.shape[:2] != (H, W):
                frame = cv2.resize(frame, (W, H))
            # mirror like a selfie camera BEFORE effects, so you see yourself naturally
            # and meme text still reads the right way round for everyone
            frame = np.ascontiguousarray(cv2.flip(frame, 1))
            now = time.perf_counter()
            seq += 1
            sig = perception.run(FramePacket(frame, now, seq), need_face=True, need_mask=comp.needs_mask())
            if sig.mask_fresh and sig.mask is not None:
                plate.update(frame, sig.mask, now)
            ctx = FxContext(frame=frame, sig=sig, now=now, assets=assets, sfx=sfx, plate=plate)

            truths = poses.update(sig, now)
            arbiter.presence_state = presence.state
            outro = comp.modal if comp.modal is not None and comp.modal.active else None
            arbiter.modal_active = outro is not None
            for it in arbiter.update(truths, (), now, busy=comp.busy_with_look(), t_on=getattr(poses, "t_on", None)):
                if it.kind == "trigger":
                    fire(it.gesture, ctx)
                elif it.kind == "release":
                    comp.release("cinema", ctx)
                elif it.kind == "cancel" and outro is not None:
                    outro.cancel(ctx)

            out = comp.render(ctx)
            dt = now - t_prev
            t_prev = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0)
            view = out.copy()
            on = [g for g, v in truths.items() if v and g != "cancel"]
            brow = getattr(poses, "debug", {}).get("eyebrows_up", {})
            brow_txt = f"brows {brow.get('level', 0):.2f}/{brow.get('threshold', 0):.2f}" if brow else ""
            hud = f"{fps:4.1f} fps | you: {presence.state} | seeing: {', '.join(on) or '-'} | {brow_txt} | sound {'on' if sound else 'off'}"
            cv2.rectangle(view, (0, H - 40), (W, H), (0, 0, 0), -1)
            cv2.putText(view, hud, (12, H - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("ZAP live demo (what Zoom sees)", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("1"):
                fire("snap1", ctx)
            elif key == ord("2"):
                fire("snap3", ctx)
            elif key == ord("3"):
                if comp.find("cinema"):
                    comp.release("cinema", ctx)
                else:
                    fire("hands_up", ctx)
            elif key == ord("4"):
                fire("cover_face", ctx)
            elif key in CUTAWAY_KEYS:
                comp.start(cutaway(CUTAWAY_KEYS[key]), ctx)
            elif key == ord("8"):
                fire("snap2", ctx)
            elif key == ord("0") and outro is not None:
                outro.cancel(ctx)
            elif key == ord("b"):
                plate.capture_clean(3.0, now)
                print("capturing a clean background: step out of view for 3 s", flush=True)
            elif key == ord("s"):
                sound = not sound
                sfx.set_enabled(sound)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        perception.close()
        sfx.close()


if __name__ == "__main__":
    main()
