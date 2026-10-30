"""Can ZAP see you? Face, hands, framing and light, in plain words for the preview's status line.

Every gesture needs the models to see a face or a hand first. In a real user's sessions the
models found no face for minutes and no hand in 95% of frames (a low camera, light behind
them) while the preview still said ARMED; nothing told them. This keeps a running picture
of what the camera sees and turns it into one instruction at a time.

Framing is measured as the room below the chin in face widths: the collar and a hand held at
the chest need about 0.8 face widths of picture under the chin (a real user had 0.26-0.45).
"""

from __future__ import annotations

from dataclasses import dataclass

from zap.signals import FACE_CHIN, Signals


@dataclass
class VisibilityParams:
    face_lost_s: float = 1.5  # no fresh face this long: "I can't see your face"
    hand_hold_s: float = 0.6  # the hand count holds this long after the last sighting
    dark_luma: float = 50.0  # the whole picture is darker than this (0-255): too dark
    face_dark: float = 75.0  # the face is darker than this ...
    backlit_ratio: float = 0.7  # ... and darker than this share of the picture: lit from behind
    room_good: float = 0.8  # face widths of picture below the chin: room for the collar and chest gestures
    room_bad: float = 0.45
    face_small: float = 0.09  # face width / frame width: too far away ...
    face_big: float = 0.42  # ... too close
    light_ema: float = 0.1


class Visibility:
    """Feed every frame with :meth:`update`; :meth:`problems` lists what to fix, most important first."""

    def __init__(self, p: VisibilityParams | None = None) -> None:
        self.p = p or VisibilityParams()
        self.face_t = -1e9
        self.ever_face = False
        self.hand_t = -1e9
        self.hands = 0
        self.room: float | None = None  # face widths of picture below the chin
        self.face_size: float | None = None  # face width / frame width
        self.luma: float | None = None
        self.face_luma: float | None = None
        self.face_frames = 0
        self.hand_frames = 0
        self.frames = 0

    def update(self, sig: Signals, now: float) -> None:
        p = self.p
        self.frames += 1
        f = sig.face
        if f is not None and f.fresh and f.width > 1.0:
            self.face_t, self.ever_face = now, True
            self.face_frames += 1
            self.room = (sig.height - float(f.lm[FACE_CHIN, 1])) / f.width
            self.face_size = f.width / sig.width
        if sig.hands:
            self.hand_t, self.hands = now, len(sig.hands)
            self.hand_frames += 1
        elif now - self.hand_t > p.hand_hold_s:
            self.hands = 0
        if sig.luma is not None:
            self.luma = sig.luma if self.luma is None else self.luma + p.light_ema * (sig.luma - self.luma)
        if sig.face_luma is not None:
            fl = sig.face_luma
            self.face_luma = fl if self.face_luma is None else self.face_luma + p.light_ema * (fl - self.face_luma)

    def face_ok(self, now: float) -> bool:
        return now - self.face_t <= self.p.face_lost_s

    def problems(self, now: float) -> list[tuple[str, str]]:
        """[(severity, instruction)], the most important first; [] = all good."""
        p = self.p
        out: list[tuple[str, str]] = []
        dark = self.luma is not None and self.luma < p.dark_luma
        if not self.face_ok(now):
            if dark:
                out.append(("error", "I can't see your face: it's too dark. Turn on a light in front of you."))
            else:
                out.append(("error", ("I can't see your face. Sit in front of the camera, face it, and keep your "
                                     "whole head in the picture.")))
            return out
        if self.face_size is not None and self.face_size < p.face_small:
            out.append(("warn", "You're far from the camera: move closer so your hands are big enough to read."))
        elif self.face_size is not None and self.face_size > p.face_big:
            out.append(("warn", "You're very close: sit back a little so your hands fit in the picture."))
        if self.room is not None and self.room < p.room_bad:
            out.append(("error", ("Your shoulders are out of the picture, so hands at your chest or collar can't "
                                 "be seen. Tilt the screen back or raise the camera, and sit back a little.")))
        elif self.room is not None and self.room < p.room_good:
            out.append(("warn", ("Little room below your chin: tilt the screen back a bit (the collar gesture "
                                "needs your shoulders in the picture).")))
        if self.face_luma is not None and self.face_luma < p.face_dark:
            if self.luma is not None and self.face_luma < p.backlit_ratio * self.luma:
                out.append(("warn", ("The light is behind you, so your face and hands are dark. Turn on a light in "
                                    "front of you or turn around.")))
            else:
                out.append(("warn", "Your face is dark: add some light in front of you."))
        elif dark:
            out.append(("warn", "It's quite dark: more light makes your hands easier to read."))
        return out

    def summary(self, now: float) -> str:
        face = "OK" if self.face_ok(now) else "not seen"
        return f"face {face}, hands {self.hands}"
