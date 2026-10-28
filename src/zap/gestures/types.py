"""Events flowing from gesture detection to the app."""

from __future__ import annotations

from dataclasses import dataclass

# Gesture names used in config [bindings] and in Intent.gesture.
SNAP1, SNAP2, SNAP3 = "snap1", "snap2", "snap3"
HANDS_UP = "hands_up"
FOREHEAD = "forehead"
COVER_FACE = "cover_face"
EYEBROWS_UP = "eyebrows_up"
COLLAR = "collar"
CANCEL = "cancel"  # open palm, only meaningful while the outro modal runs
THUMBS_UP = "thumbs_up"  # confirm, only meaningful in zoom.confirm = "thumbs_up" mode
TWO_THUMBS = "two_thumbs_up"  # both hands thumbs-up: "deal closed"
PANIC = "panic"  # Intent kind: stop every effect, show the plain camera (Ctrl+Alt+P)

POSE_GESTURES = (HANDS_UP, TWO_THUMBS, FOREHEAD, COVER_FACE, EYEBROWS_UP, COLLAR)
ALL_GESTURES = (SNAP1, SNAP2, SNAP3, *POSE_GESTURES)


@dataclass(frozen=True)
class SnapBurst:
    """A completed group of 1-3 finger snaps."""

    count: int
    t_first: float  # perf_counter time of the first snap onset
    t_last: float
    where: tuple[float, float] | None = None  # output-pixel position of the snapping hand, if known


@dataclass(frozen=True)
class Intent:
    """What the arbiter decided this frame.

    kind:
      "trigger" - start the effect bound to `gesture`
      "release" - the held gesture ended (hold-to-show effects like cinema fade out)
      "cancel"  - cancel the running modal (outro)
      "confirm" - confirm the running modal (thumbs_up confirm mode)
      "charge"  - first snap of a possible burst: show the ember pre-roll at `where`
      "discharge" - the pre-roll should fade (burst discarded)
      "panic"   - stop every effect, show the plain camera (Ctrl+Alt+P)
    """

    kind: str
    gesture: str = ""
    t: float = 0.0
    where: tuple[float, float] | None = None
