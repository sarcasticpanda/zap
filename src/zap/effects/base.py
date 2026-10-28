"""The effect API every meme effect implements.

The compositor draws layers in this order each frame::

    PRESENCE  (you: visible / dissolving / vanished / reassembling)
    LOOK      (one full-frame grade at a time: cinema, sad)
    OVERLAY   (stackable sprites: money rain, charge sparks, stamps)
    CUTAWAY   (replaces the whole frame with a meme image or clip)
    MODAL     (the end-meeting outro, above everything)

An effect draws into ``out`` in place and clears ``self.active`` when it is done.
Effects never block: anything slow is prepared in ``start`` or ahead of time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from zap.render.blend import upsample_mask
from zap.signals import Signals

if TYPE_CHECKING:
    from zap.assets import AssetResolver
    from zap.effects.plate import BackgroundPlate


class Layer(IntEnum):
    PRESENCE = 0
    LOOK = 1
    OVERLAY = 2
    CUTAWAY = 3
    MODAL = 4


class SfxSink(Protocol):
    def play(self, key: str) -> None: ...
    def stop_all(self) -> None: ...


class NullSfx:
    def play(self, key: str) -> None:
        pass

    def stop_all(self) -> None:
        pass


@dataclass
class FxContext:
    """Per-frame view of the world handed to effects."""

    frame: np.ndarray  # live camera frame, BGR uint8 at output size; do not modify
    sig: Signals
    now: float
    assets: AssetResolver | None = None
    sfx: SfxSink = field(default_factory=NullSfx)
    plate: BackgroundPlate | None = None
    _mask_cache: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def W(self) -> int:
        return self.frame.shape[1]

    @property
    def H(self) -> int:
        return self.frame.shape[0]

    def person_mask(self, blur: int = 0) -> np.ndarray | None:
        """Person confidence as float32 at output size (cached per blur level), or None."""
        if self.sig.mask is None:
            return None
        if blur not in self._mask_cache:
            self._mask_cache[blur] = upsample_mask(self.sig.mask, self.W, self.H, blur)
        return self._mask_cache[blur]


class Effect:
    name = "effect"
    layer = Layer.OVERLAY
    needs_mask = False  # perception should compute the person mask every frame while active
    needs_face = False  # perception should run the face model every frame while active
    sfx_key: str | None = None  # sound played on start, looked up in assets/sfx

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.active = False
        self.t0 = 0.0

    def start(self, ctx: FxContext, **kw: Any) -> None:
        self.active = True
        self.t0 = ctx.now
        if self.sfx_key:
            ctx.sfx.play(self.sfx_key)

    def release(self, ctx: FxContext) -> None:
        """The gesture that holds this effect ended (only matters for hold-to-show effects)."""

    def update(self, ctx: FxContext, out: np.ndarray) -> None:
        raise NotImplementedError

    def abort(self) -> None:
        self.active = False

    def elapsed(self, ctx: FxContext) -> float:
        return ctx.now - self.t0
