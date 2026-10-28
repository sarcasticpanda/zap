"""``zap`` command line: run ZAP, check the PC, fetch models, and the tools around them.

    zap                      same as "zap run"
    zap run [--seconds N] [--no-preview] [--armed] [--snap-entry] [--no-mic]
    zap doctor [--zoom]      check this PC and print how to fix problems
    zap fetch-models         download the MediaPipe models
    zap bench [--seconds N]  measure perception speed on this laptop
    zap try EFFECT           live pipeline, triggering EFFECT every few seconds (cinema, presence, cutaway:jim, snap3...)
    zap camtest              camera -> virtual camera only, with fps
    zap record [--seconds N] save landmarks + mic + a small video for tuning (recordings/<time>/)
    zap replay DIR           run a recording through the gesture detectors and print the events

Exit codes: 0 ok, 1 crash, 2 setup problem (camera, virtual camera, config, models), 3 already running.
"""

from __future__ import annotations

import argparse
import difflib
import logging
import os
import sys

from zap import __version__

os.environ.setdefault("GLOG_minloglevel", "2")  # before anything imports mediapipe

log = logging.getLogger("zap")

COMMANDS = ("run", "doctor", "fetch-models", "bench", "try", "camtest", "record", "replay")


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="zap", description="Gesture-triggered meme effects for Zoom.")
    ap.add_argument("--version", action="version", version=f"zap {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    def live(p: argparse.ArgumentParser, seconds_help: str = "exit after this many seconds") -> None:
        p.add_argument("--seconds", type=float, default=None, help=seconds_help)
        p.add_argument("--no-preview", action="store_true", help="no local preview window")
        p.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    run = sub.add_parser("run", help="start ZAP (default)")
    live(run)
    run.add_argument("--armed", action="store_true", help="start with gestures armed")
    run.add_argument("--snap-entry", action="store_true", help="start invisible; snap once to appear")
    run.add_argument("--no-mic", action="store_true", help="don't use the microphone (single snaps only)")

    doctor = sub.add_parser("doctor", help="check this PC is ready")
    doctor.add_argument("--zoom", action="store_true", help="also list Zoom's windows and buttons")

    fetch = sub.add_parser("fetch-models", help="download the MediaPipe models")
    fetch.add_argument("--force", action="store_true", help="download again even if present")
    fetch.add_argument("--faceswap", action="store_true",
                       help="also download the face-swap models (~450 MB, non-commercial license) for your face on Harvey")

    bench = sub.add_parser("bench", help="measure perception speed")
    bench.add_argument("--seconds", type=float, default=20.0)

    tr = sub.add_parser("try", help="trigger one effect every few seconds on the live camera")
    tr.add_argument("effect", help="effect or gesture: presence, cinema, money, sad, outro, cutaway:<name>, snap1...")
    live(tr)
    tr.add_argument("--every", type=float, default=4.0, help="seconds between triggers")

    cam = sub.add_parser("camtest", help="camera -> virtual camera, no effects")
    live(cam)

    rec = sub.add_parser("record", help="record landmarks, mic and a small video for tuning")
    rec.add_argument("--seconds", type=float, default=30.0)
    rec.add_argument("--no-preview", action="store_true")
    rec.add_argument("-v", "--verbose", action="store_true")

    rep = sub.add_parser("replay", help="run a recording through the gesture detectors")
    rep.add_argument("dir", help="a folder made by 'zap record'")
    return ap


def _load_config():
    from zap import config

    return config.load()


def _valid_target(name: str, cfg) -> str | None:
    """None if ``name`` can be tried, else an error message."""
    from zap.config import EFFECT_TARGETS
    from zap.gestures.types import ALL_GESTURES

    if name in ALL_GESTURES or (name in EFFECT_TARGETS and name != "none"):
        return None
    if name.startswith("cutaway:"):
        cut = name.split(":", 1)[1]
        if cut in cfg.cutaways:
            return None
        return f"unknown cutaway '{cut}' (configured: {', '.join(cfg.cutaways) or 'none'})"
    options = [*ALL_GESTURES, *(t for t in EFFECT_TARGETS if t != "none"), *(f"cutaway:{c}" for c in cfg.cutaways)]
    hint = difflib.get_close_matches(name, options, n=1)
    return f"unknown effect '{name}'" + (f" (did you mean '{hint[0]}'?)" if hint else "") + \
        f"; choose from {', '.join(options)}"


def _run_app(args, try_target: str | None = None) -> int:
    from zap.app import App, RunOptions, setup_logging

    setup_logging(getattr(args, "verbose", False))
    cfg = _load_config()
    if try_target is not None:
        problem = _valid_target(try_target, cfg)
        if problem:
            print(f"zap try: {problem}", file=sys.stderr)
            return 2
    opts = RunOptions(
        preview=False if args.no_preview else None,
        seconds=args.seconds,
        armed=getattr(args, "armed", False) or None,
        snap_entry=True if getattr(args, "snap_entry", False) else None,
        try_target=try_target,
        try_every_s=getattr(args, "every", 4.0),
        use_mic=not getattr(args, "no_mic", False),
    )
    return App(cfg, opts).run()


def _dispatch(args) -> int:
    cmd = args.cmd
    if cmd == "doctor":
        from zap import doctor

        return doctor.run(zoom=args.zoom)
    if cmd == "fetch-models":
        from zap import fetch_models

        fetch_models.fetch(force=args.force)
        if args.faceswap:
            fetch_models.fetch_faceswap(force=args.force)
        return 0
    if cmd == "bench":
        from zap import bench

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        bench.run(seconds=args.seconds)
        return 0
    if cmd == "camtest":
        from zap.app import run_camtest, setup_logging

        setup_logging(args.verbose)
        return run_camtest(_load_config(), args.seconds, preview=not args.no_preview)
    if cmd == "record":
        from zap.app import acquire_single_instance, release_single_instance, setup_logging
        from zap.recording import record

        setup_logging(args.verbose)
        mutex = acquire_single_instance()
        try:
            folder = record(_load_config(), seconds=args.seconds, preview=not args.no_preview)
        finally:
            release_single_instance(mutex)
        print(f"recorded into {folder}\nreplay it with: zap replay \"{folder}\"")
        return 0
    if cmd == "replay":
        from zap.recording import replay

        print(replay(args.dir).format())
        return 0
    if cmd == "try":
        return _run_app(args, try_target=args.effect)
    return _run_app(args)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0] not in COMMANDS and argv[0].startswith("-") and argv[0] not in ("-h", "--help",
                                                                                            "--version")):
        argv.insert(0, "run")
    args = _parser().parse_args(argv)
    if args.cmd is None:
        args = _parser().parse_args(["run", *argv])

    from zap.app import AlreadyRunning, ModelsMissing
    from zap.config import ConfigError
    from zap.devices.camera import CameraError
    from zap.devices.vcam import VirtualCamError

    try:
        return _dispatch(args)
    except ConfigError as e:
        print(f"Problem in config.toml: {e}\nFix the file (or delete the line) and run zap again.", file=sys.stderr)
        return 2
    except (CameraError, VirtualCamError) as e:
        print(f"{e}\nRun 'zap doctor' for a full check.", file=sys.stderr)
        return 2
    except ModelsMissing as e:
        print(str(e), file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        if "model" in str(e).lower():
            print(f"{e}\nRun: zap fetch-models", file=sys.stderr)
            return 2
        print(str(e), file=sys.stderr)
        return 2
    except AlreadyRunning as e:
        print(str(e), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
