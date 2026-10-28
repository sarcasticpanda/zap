"""A clean plate of the room behind the user.

Vanishing needs to show what is behind you. The app feeds frames that come
with a fresh person mask; pixels the mask calls background are blended slowly
into a float plate, which averages out sensor noise and follows lighting
changes. Pixels never seen (where you have always been sitting) are inpainted
at 160x90 about once a second: blurry, but plausible behind a silhouette that
keeps moving. Masks are thresholded and dilated at half the mask resolution
and upsampled once, so one update is about two full-frame cv2 passes.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

ALPHA = 0.05  # per-update blend weight for background pixels
CAPTURE_ALPHA = 0.1  # weight while capturing a clean plate (user stepped out)
PERSON_THRESHOLD = 0.2  # person confidence at or above this is never plate
DILATE_PX = 7  # dilation radius for a 640-wide mask (~25 px at 720p); scaled with mask width
FILL_INTERVAL_S = 1.0
FILL_SIZE = (160, 90)  # inpaint resolution (w, h)
LOAD_TOLERANCE = 0.15  # saved plate must be within 15% of the live mean luminance


def _luma(img: np.ndarray) -> float:
    b, g, r, _ = cv2.mean(img)
    return 0.114 * b + 0.587 * g + 0.299 * r


class BackgroundPlate:
    """Running estimate of the empty room at output resolution (BGR)."""

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self._plate = np.zeros((h, w, 3), np.float32)
        self._u8 = np.zeros((h, w, 3), np.uint8)
        self._bg_full = np.zeros((h, w), np.uint8)
        self._dirty = False
        self._seen: np.ndarray | None = None  # uint8 0/255 at half the mask resolution
        self._seen_all = False  # whole frame known (loaded or captured)
        self._seen_any = False
        self._last_small: np.ndarray | None = None  # last frame at FILL_SIZE, for the fallback
        self._fallback: np.ndarray | None = None
        self._last_fill: float | None = None
        self._now = 0.0
        self._capture_t0: float | None = None
        self._capture_until: float | None = None

    # ------------------------------------------------------------ state
    @property
    def coverage(self) -> float:
        """Fraction of the frame seen as background at least once (0..1)."""
        if self._seen_all:
            return 1.0
        if self._seen is None:
            return 0.0
        return cv2.countNonZero(self._seen) / self._seen.size

    @property
    def capturing(self) -> bool:
        return self._capture_until is not None and self._now < self._capture_until

    @property
    def progress(self) -> float:
        """Clean-capture progress 0..1 (0 when no capture was requested)."""
        if self._capture_t0 is None or self._capture_until is None:
            return 0.0
        span = max(1e-6, self._capture_until - self._capture_t0)
        return min(1.0, max(0.0, (self._now - self._capture_t0) / span))

    @property
    def ready(self) -> bool:
        return self._seen_any

    def reset(self) -> None:
        self.__init__(self.w, self.h)

    # ------------------------------------------------------------ feeding
    def capture_clean(self, duration: float = 3.0, now: float = 0.0) -> None:
        """Learn the whole frame for `duration` seconds, ignoring the mask (the user stepped out).

        Call update() every frame while `capturing` (the mask may be None then).
        """
        self._now = now
        self._capture_t0 = now
        self._capture_until = now + duration
        log.info("capturing a clean plate for %.1f s", duration)

    def update(self, frame: np.ndarray, mask: np.ndarray | None, now: float) -> None:
        """Blend background pixels of `frame` (BGR uint8, output size) into the plate.

        `mask` is the person confidence at inference resolution (float32, HxW or HxWx1).
        """
        self._now = now
        if frame.shape[:2] != (self.h, self.w):
            frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
        if not self._seen_any:  # only the pre-learning fallback needs it
            self._last_small = cv2.resize(frame, FILL_SIZE, interpolation=cv2.INTER_AREA)
            self._fallback = None
        if self.capturing:
            self._capture(frame)
            return
        if mask is None:
            return
        m = mask[..., 0] if mask.ndim == 3 else mask
        bg = self._background_mask(m)
        if self._seen is None or self._seen.shape != bg.shape:
            self._seen = np.full_like(bg, 255 if self._seen_all else 0)
        new = cv2.bitwise_and(bg, cv2.bitwise_not(self._seen))
        cv2.bitwise_or(self._seen, bg, dst=self._seen)
        cv2.resize(bg, (self.w, self.h), dst=self._bg_full, interpolation=cv2.INTER_NEAREST)
        if cv2.countNonZero(new):
            # first sighting: take the pixel as is instead of fading in from nothing
            new_full = cv2.resize(new, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
            cv2.accumulateWeighted(frame, self._plate, 1.0, mask=new_full)
            self._seen_any = True
        cv2.accumulateWeighted(frame, self._plate, ALPHA, mask=self._bg_full)
        self._dirty = True
        if self._seen_any and (self._last_fill is None or now - self._last_fill >= FILL_INTERVAL_S):
            self._fill_holes()
            self._last_fill = now

    def _capture(self, frame: np.ndarray) -> None:
        alpha = CAPTURE_ALPHA if self._seen_any else 1.0
        cv2.accumulateWeighted(frame, self._plate, alpha)
        self._seen_all = self._seen_any = True
        if self._seen is not None:
            self._seen[:] = 255
        self._dirty = True

    def _background_mask(self, m: np.ndarray) -> np.ndarray:
        """uint8 255 = background, at half the mask resolution (dilating there is ~4x cheaper)."""
        person = cv2.compare(m, PERSON_THRESHOLD, cv2.CMP_GE)
        hw, hh = max(1, m.shape[1] // 2), max(1, m.shape[0] // 2)
        person = cv2.resize(person, (hw, hh), interpolation=cv2.INTER_AREA)
        cv2.threshold(person, 0, 255, cv2.THRESH_BINARY, dst=person)  # any person in the cell counts
        r = max(1, math.ceil(DILATE_PX * hw / 640))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        return cv2.bitwise_not(cv2.dilate(person, kernel))

    def _fill_holes(self) -> None:
        """Inpaint never-seen pixels at low resolution and paste them into the holes only."""
        if self._seen_all or self._seen is None:
            return
        hole = cv2.bitwise_not(self._seen)
        if not cv2.countNonZero(hole):
            return
        small = cv2.resize(self._image_u8(), FILL_SIZE, interpolation=cv2.INTER_AREA)
        # any cell touching a hole counts as hole, so unseen values never act as a source
        hole_small = cv2.resize(hole, FILL_SIZE, interpolation=cv2.INTER_AREA)
        cv2.threshold(hole_small, 0, 255, cv2.THRESH_BINARY, dst=hole_small)
        if cv2.countNonZero(hole_small) >= hole_small.size:
            return
        filled = cv2.inpaint(small, hole_small, 3, cv2.INPAINT_TELEA)
        up = cv2.resize(filled, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        hole_full = cv2.resize(hole, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        cv2.accumulateWeighted(up, self._plate, 1.0, mask=hole_full)
        self._dirty = True

    # ------------------------------------------------------------ reading
    def _image_u8(self) -> np.ndarray:
        if self._dirty:
            cv2.convertScaleAbs(self._plate, dst=self._u8)
            self._dirty = False
        return self._u8

    def image(self) -> np.ndarray:
        """The plate as BGR uint8 at output size. Shared buffer: read it, do not modify it.

        Before any background has been seen this is the last frame, heavily blurred.
        """
        if self._seen_any:
            return self._image_u8()
        if self._fallback is None:
            if self._last_small is None:
                return self._u8
            blurred = cv2.GaussianBlur(self._last_small, (0, 0), 6)
            self._fallback = cv2.resize(blurred, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return self._fallback

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> bool:
        if not self._seen_any:
            log.warning("plate not saved: nothing learned yet")
            return False
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(path.suffix or ".png", self._image_u8())
        if not ok:
            return False
        buf.tofile(str(path))  # tofile/imdecode handle non-ASCII paths, unlike imwrite on Windows
        return True

    def load(self, path: Path, current_frame: np.ndarray) -> bool:
        """Load a saved plate if its mean luminance is within 15% of the current frame's."""
        path = Path(path)
        if not path.is_file():
            return False
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            log.warning("could not read saved plate %s", path.name)
            return False
        if img.shape[:2] != (self.h, self.w):
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        lp, lc = _luma(img), _luma(current_frame)
        if abs(lp - lc) > LOAD_TOLERANCE * max(lc, 1.0):
            log.info("saved plate ignored: lighting changed (luma %.0f vs %.0f now)", lp, lc)
            return False
        self._plate[:] = img
        self._seen_all = self._seen_any = True
        if self._seen is not None:
            self._seen[:] = 255
        self._dirty = True
        return True
