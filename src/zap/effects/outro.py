"""The "To Be Continued" outro that ends your meeting.

A double snap freezes the outgoing picture JoJo-style (sepia, an arrow sliding
in), counts down 3-2-1 and then asks the app to leave Zoom. Leaving is the one
irreversible thing ZAP does, so the countdown is cancellable right up to zero,
``on_finish`` fires exactly once and only when the countdown completes, and in
"thumbs_up" mode a missing confirmation cancels rather than leaves.

The freeze is taken from ``out`` on the first counting frame, i.e. exactly what
the meeting was seeing (lower layers already composited), and the sepia frame
is computed once, so holding it costs a single copy per frame.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from zap.effects.base import Effect, FxContext, Layer
from zap.effects.cutaway import blit_scaled
from zap.effects.registry import register
from zap.render.blend import (
    blit_premult,
    clamp01,
    ease_in_out_sine,
    ease_out_back,
    ease_out_cubic,
    make_premult,
    vignette_map,
)
from zap.render.text import TextStyle, load_font, render_emoji, render_text

log = logging.getLogger(__name__)

CONFIRMING, COUNTING, LEAVING, CANCELLED, DONE = "confirming", "counting", "leaving", "cancelled", "done"

# Standard sepia matrix rearranged for BGR in and out.
SEPIA_BGR = np.float32([
    [0.131, 0.534, 0.272],
    [0.168, 0.686, 0.349],
    [0.189, 0.769, 0.393],
])
TRANSITION_S = 0.4  # colour -> sepia
THUNK_S = 0.14  # the freeze "thunk": a small punch and flash
ARROW_IN_S = 0.6
POP_S = 0.22  # each numeral pops in over this long
CANCEL_S = 0.6  # the "jk" wipe back to live

ARROW_FILL = (176, 158, 88, 255)  # olive/khaki, RGBA
ARROW_EDGE = (66, 46, 20, 255)  # dark brown
ARROW_FONT = "georgiaz.ttf"  # Georgia Bold Italic
NUMERAL_FONT = "impact.ttf"
PROMPT_FONT = "segoeuib.ttf"
THUMBS_UP, RAISED_HAND, SWEAT_SMILE = "\U0001F44D", "✋", "\U0001F605"


@register("outro")
class Outro(Effect):
    """Freeze, sepia, "To Be Continued" arrow, 3-2-1, then ``on_finish()`` once.

    ``confirm``: "countdown" starts counting at once; "thumbs_up" first waits up
    to ``confirm_timeout_s`` for :meth:`confirm` and cancels on timeout.
    ``on_finish`` runs on the render thread inside ``update``; it must only
    hand the leave off to a worker thread and return.
    """

    layer = Layer.MODAL
    sfx_key = "outro"

    def __init__(
        self,
        countdown_s: float = 3.0,
        confirm: str = "countdown",
        on_finish: Callable[[], Any] | None = None,
        confirm_timeout_s: float = 5.0,
        hold_s: float = 10.0,
        **params: Any,
    ) -> None:
        if confirm not in ("countdown", "thumbs_up"):
            raise ValueError(f"confirm must be 'countdown' or 'thumbs_up', got {confirm!r}")
        super().__init__(countdown_s=countdown_s, confirm=confirm, on_finish=on_finish,
                         confirm_timeout_s=confirm_timeout_s, hold_s=hold_s, **params)
        self.countdown_s = max(0.1, float(countdown_s))
        self.confirm_mode = confirm
        self.on_finish = on_finish
        self.confirm_timeout_s = float(confirm_timeout_s)
        self.hold_s = float(hold_s)
        self._reset()

    def _reset(self) -> None:
        self._state = DONE
        self._finished = False  # on_finish has been called
        self._t_count: float | None = None
        self._t_finish = 0.0
        self._t_cancel: float | None = None
        self._confirm_pending = False
        self._cancel_pending = False
        self._last_tc = 0.0
        self._seconds_left = self.countdown_s
        self._freeze: np.ndarray | None = None
        self._sepia: np.ndarray | None = None
        self._buf: np.ndarray | None = None
        self._wipe_src: np.ndarray | None = None

    @classmethod
    def prewarm(cls, width: int, height: int, countdown_s: float = 3.0) -> None:
        """Build the vignette and sprites ahead of time so the freeze frame doesn't hitch (~50 ms cold)."""
        _sepia_vignette(width, height)
        _arrow_sprite(width)
        _prompt_sprite(width)
        _jk_sprite(width)
        for n in range(1, max(1, round(countdown_s)) + 1):
            _numeral_sprite(n, width)

    # ------------------------------------------------------------ HUD
    @property
    def state(self) -> str:
        """"confirming" | "counting" | "leaving" | "cancelled" | "done"."""
        return self._state

    @property
    def seconds_left(self) -> float:
        """Countdown seconds remaining (full while confirming, 0 once leaving)."""
        return self._seconds_left

    @property
    def finished(self) -> bool:
        """True once on_finish has been called."""
        return self._finished

    # ------------------------------------------------------------ control
    def start(self, ctx: FxContext, **kw: Any) -> None:
        self._reset()
        self.active = True
        self.t0 = ctx.now
        if self.confirm_mode == "thumbs_up":
            self._state = CONFIRMING
        else:
            self._begin_count(ctx)

    def confirm(self, ctx: FxContext | None = None) -> bool:
        """Thumbs-up received. Returns False if there was nothing to confirm."""
        if not self.active or self._state != CONFIRMING:
            return False
        if ctx is not None:
            self._begin_count(ctx)
        else:
            self._confirm_pending = True
        return True

    def cancel(self, ctx: FxContext | None = None) -> bool:
        """Stop now and wipe back to live. Returns True if this prevented on_finish.

        Too late once leaving: on_finish already ran, so a "jk" would lie.
        """
        if not self.active or self._state in (CANCELLED, DONE, LEAVING):
            return False
        self._state = CANCELLED
        self._confirm_pending = False
        self._seconds_left = 0.0
        if ctx is not None:
            self._t_cancel = ctx.now
            ctx.sfx.stop_all()
        else:
            self._cancel_pending = True
        return not self._finished

    def abort(self) -> None:
        self.active = False
        if self._state != CANCELLED:
            self._state = DONE

    def _begin_count(self, ctx: FxContext) -> None:
        self._state = COUNTING
        self._t_count = ctx.now
        self._confirm_pending = False
        self._freeze = None  # captured from `out` on the next update
        ctx.sfx.play(self.sfx_key)

    # ------------------------------------------------------------ render
    def update(self, ctx: FxContext, out: np.ndarray) -> None:
        now = ctx.now
        if self._cancel_pending:
            self._cancel_pending = False
            self._t_cancel = now
            ctx.sfx.stop_all()
        if self._state == CONFIRMING:
            if self._confirm_pending:
                self._begin_count(ctx)
            elif now - self.t0 >= self.confirm_timeout_s:
                log.info("outro: no thumbs-up within %.1f s; cancelled", self.confirm_timeout_s)
                self.cancel(ctx)
            else:
                self._seconds_left = self.countdown_s
                self._draw_prompt(out, now - self.t0)
                return
        if self._state in (COUNTING, LEAVING):
            self._update_modal(ctx, out)
        elif self._state == CANCELLED:
            self._draw_cancel(out, now - (self._t_cancel if self._t_cancel is not None else now))

    def _update_modal(self, ctx: FxContext, out: np.ndarray) -> None:
        now = ctx.now
        if self._freeze is None:
            self._capture(out)
        tc = now - self._t_count
        if self._state == COUNTING and tc >= self.countdown_s:
            self._state = LEAVING
            self._t_finish = now
            self._fire()
        if self._state == LEAVING and now - self._t_finish >= self.hold_s:
            self.active = False
            self._state = DONE
            return
        self._seconds_left = max(0.0, self.countdown_s - tc) if self._state == COUNTING else 0.0
        self._last_tc = tc
        self._draw_modal(out, tc)

    def _fire(self) -> None:
        self._finished = True  # set first: on_finish must never run twice
        if self.on_finish is None:
            log.info("outro finished (no on_finish handler)")
            return
        try:
            self.on_finish()
        except Exception:
            log.exception("outro on_finish handler failed")

    def _capture(self, out: np.ndarray) -> None:
        H, W = out.shape[:2]
        self._freeze = out.copy()
        sepia = cv2.transform(self._freeze, SEPIA_BGR)
        self._sepia = cv2.multiply(sepia, _sepia_vignette(W, H), scale=1 / 255)
        self._buf = np.empty_like(out)

    def _draw_modal(self, dst: np.ndarray, tc: float) -> None:
        """The frozen frame, arrow and numeral at `tc` seconds into the countdown. Overwrites dst."""
        H, W = dst.shape[:2]
        if tc < TRANSITION_S:
            k = ease_out_cubic(tc / TRANSITION_S)
            cv2.addWeighted(self._freeze, 1 - k, self._sepia, k, 0, dst=self._buf)
            s = 1.0 + 0.05 * (1.0 - ease_out_cubic(tc / THUNK_S))
            if s > 1.001:
                M = np.float32([[s, 0, (1 - s) * W / 2], [0, s, (1 - s) * H / 2]])
                cv2.warpAffine(self._buf, M, (W, H), dst=dst, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
            else:
                np.copyto(dst, self._buf)
            if tc < 0.08:
                flash = 70 * (1 - tc / 0.08)
                cv2.add(dst, (flash, flash, flash, 0), dst=dst)
        else:
            np.copyto(dst, self._sepia)

        k = W / 1280
        pm, inv3 = _arrow_sprite(W)
        x_end, x_start = int(36 * k), -pm.shape[1] - 10
        x = int(x_start + (x_end - x_start) * ease_out_back(tc / ARROW_IN_S))
        blit_premult(dst, pm, inv3, x, H - pm.shape[0] - int(40 * k))

        if self._state == COUNTING and tc < self.countdown_s:
            n = max(1, round(self.countdown_s))
            step = self.countdown_s / n
            i = min(n - 1, int(tc // step))
            local = tc - i * step
            pm, inv3 = _numeral_sprite(n - i, W)
            scale = 0.4 + 0.6 * ease_out_back(local / POP_S, overshoot=2.2)
            blit_scaled(dst, pm, inv3, W * 0.78, H * 0.45, scale, local / 0.06)

    def _draw_prompt(self, out: np.ndarray, t: float) -> None:
        H, W = out.shape[:2]
        pm, inv3 = _prompt_sprite(W)
        pop = 0.85 + 0.15 * ease_out_back(t / 0.25)
        cx, cy = W / 2, H - pm.shape[0] / 2 - int(48 * W / 1280)
        blit_scaled(out, pm, inv3, cx, cy, pop, t / 0.12)
        # time left to confirm, as a shrinking bar under the pill
        left = clamp01(1.0 - t / self.confirm_timeout_s)
        bw, bh = int(pm.shape[1] * 0.8), max(2, int(5 * W / 1280))
        x0, y0 = int(cx - bw / 2), int(cy + pm.shape[0] / 2 + 8 * W / 1280)
        if left > 0 and y0 + bh <= H:
            cv2.rectangle(out, (x0, y0), (x0 + int(bw * left), y0 + bh), (255, 255, 255), -1)

    def _draw_cancel(self, out: np.ndarray, tw: float) -> None:
        if tw >= CANCEL_S:
            self.active = False
            return
        H, W = out.shape[:2]
        if self._freeze is not None:
            if self._wipe_src is None:
                self._wipe_src = np.empty_like(out)
                self._draw_modal(self._wipe_src, self._last_tc)
            # live (already in `out`) is revealed left to right behind a bright edge
            x = int(W * ease_in_out_sine(tw / CANCEL_S))
            if x < W:
                np.copyto(out[:, x:], self._wipe_src[:, x:])
                e = max(2, int(6 * W / 1280))
                out[:, max(0, x - e):min(W, x + e)] = 255
        pm, inv3 = _jk_sprite(W)
        scale = 0.6 + 0.4 * ease_out_back(tw / 0.18)
        fade = clamp01((CANCEL_S - tw) / 0.2)
        blit_scaled(out, pm, inv3, W / 2, H / 2, scale, fade)


# ---------------------------------------------------------------- sprites
@lru_cache(maxsize=2)
def _sepia_vignette(W: int, H: int) -> np.ndarray:
    return vignette_map(W, H, strength=0.45, radius=0.9)


@lru_cache(maxsize=4)
def _arrow_sprite(W: int) -> tuple[np.ndarray, np.ndarray]:
    """The JoJo "To Be Continued" arrow, pointing left, drawn supersampled for clean edges."""
    k, ss = W / 1280, 3
    w, h = max(40, int(520 * k)), max(10, int(104 * k))
    w2, h2 = w * ss, h * ss
    img = Image.new("RGBA", (w2, h2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lw = max(2, int(5 * k)) * ss
    head = int(0.19 * w2)
    sy0, sy1 = int(h2 * 0.2), int(h2 * 0.8)
    pts = [(lw, h2 / 2), (head, lw), (head, sy0), (w2 - lw, sy0), (w2 - lw, sy1), (head, sy1), (head, h2 - lw)]
    d.polygon(pts, fill=ARROW_FILL, outline=ARROW_EDGE, width=lw)
    font = load_font(ARROW_FONT, max(8, int(44 * k * ss)))
    d.text(((head + w2) / 2, h2 / 2), "To Be Continued", font=font, fill=ARROW_EDGE, anchor="mm")
    rgba = np.asarray(img.resize((w, h), Image.LANCZOS)).copy()
    return make_premult(rgba)


@lru_cache(maxsize=16)
def _numeral_sprite(n: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    k = W / 1280
    style = TextStyle(font=NUMERAL_FONT, size=max(24, int(250 * k)), color=(255, 255, 255, 255),
                      stroke=max(2, int(11 * k)), stroke_color=(0, 0, 0, 255), shadow=max(2, int(10 * k)))
    return make_premult(render_text(str(n), style))


@lru_cache(maxsize=4)
def _prompt_sprite(W: int) -> tuple[np.ndarray, np.ndarray]:
    """"👍 to leave · ✋ to cancel" on a dark pill."""
    k = W / 1280
    size = max(12, int(38 * k))
    style = TextStyle(font=PROMPT_FONT, size=size, color=(255, 255, 255, 255))
    parts = [render_emoji(THUMBS_UP, int(size * 1.3)), render_text(" to leave  ·  ", style),
             render_emoji(RAISED_HAND, int(size * 1.3)), render_text(" to cancel", style)]
    row = _hstack_rgba(parts)
    pad_x, pad_y = int(30 * k), int(14 * k)
    pill = Image.new("RGBA", (row.shape[1] + 2 * pad_x, row.shape[0] + 2 * pad_y), (0, 0, 0, 0))
    ImageDraw.Draw(pill).rounded_rectangle((0, 0, pill.width - 1, pill.height - 1),
                                           radius=pill.height // 2, fill=(12, 12, 16, 185))
    pill.alpha_composite(Image.fromarray(row), (pad_x, pad_y))
    return make_premult(np.asarray(pill).copy())


@lru_cache(maxsize=4)
def _jk_sprite(W: int) -> tuple[np.ndarray, np.ndarray]:
    k = W / 1280
    style = TextStyle(font=NUMERAL_FONT, size=max(24, int(170 * k)), color=(255, 255, 255, 255),
                      stroke=max(2, int(9 * k)), stroke_color=(0, 0, 0, 255), shadow=max(2, int(8 * k)))
    text = render_text("jk", style)
    gap = np.zeros((text.shape[0], max(4, int(24 * k)), 4), np.uint8)
    return make_premult(_hstack_rgba([text, gap, render_emoji(SWEAT_SMILE, int(text.shape[0] * 0.8))]))


def _hstack_rgba(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate RGBA sprites left to right, vertically centred."""
    h = max(p.shape[0] for p in parts)
    row = np.zeros((h, sum(p.shape[1] for p in parts), 4), np.uint8)
    x = 0
    for p in parts:
        y = (h - p.shape[0]) // 2
        row[y:y + p.shape[0], x:x + p.shape[1]] = p
        x += p.shape[1]
    return row

