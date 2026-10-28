"""Text and emoji sprites rendered with Pillow and cached.

Rendering text every frame is wasteful, so every (text, style) combination is
drawn once into an RGBA array and reused. Colour emoji come from Segoe UI Emoji
(``embedded_color=True``); only single-codepoint emoji are reliable because
Pillow on Windows has no raqm shaping.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from zap.paths import font_path

FALLBACK_FONTS = ("arialbd.ttf", "arial.ttf", "segoeui.ttf")
EMOJI_FONT = "seguiemj.ttf"


@lru_cache(maxsize=64)
def load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for candidate in (name, *FALLBACK_FONTS):
        p = font_path(candidate)
        if p is not None:
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size)


@dataclass(frozen=True)
class TextStyle:
    font: str = "impact.ttf"
    size: int = 64
    color: tuple[int, int, int, int] = (255, 255, 255, 255)  # RGBA
    stroke: int = 0
    stroke_color: tuple[int, int, int, int] = (0, 0, 0, 255)
    tracking: float = 0.0  # extra letter spacing as a fraction of the font size
    shadow: int = 0  # blur radius of a soft drop shadow, 0 = none
    shadow_alpha: float = 0.5


def _measure(font: ImageFont.FreeTypeFont, text: str, style: TextStyle) -> tuple[int, int, int, int]:
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    if style.tracking <= 0:
        return probe.textbbox((0, 0), text, font=font, stroke_width=style.stroke)
    x = 0.0
    top, bottom = 10**9, -(10**9)
    for ch in text:
        _, t, _, b = probe.textbbox((0, 0), ch, font=font, stroke_width=style.stroke)
        top, bottom = min(top, t), max(bottom, b)
        x += font.getlength(ch) + style.tracking * style.size
    x -= style.tracking * style.size
    return 0, top, round(x) + 2 * style.stroke, bottom


@lru_cache(maxsize=256)
def render_text(text: str, style: TextStyle = TextStyle()) -> np.ndarray:  # noqa: B008 - frozen, hashable
    """Render text to a tightly cropped RGBA uint8 array (Pillow channel order)."""
    font = load_font(style.font, style.size)
    l, t, r, b = _measure(font, text, style)
    pad = style.stroke + style.shadow * 2 + 2
    w, h = (r - l) + 2 * pad, (b - t) + 2 * pad
    img = Image.new("RGBA", (max(1, w), max(1, h)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    origin = (pad - l, pad - t)

    def draw_text(d: ImageDraw.ImageDraw, fill, stroke_fill) -> None:
        if style.tracking <= 0:
            d.text(origin, text, font=font, fill=fill, stroke_width=style.stroke, stroke_fill=stroke_fill)
            return
        x, y = origin
        for ch in text:
            d.text((x, y), ch, font=font, fill=fill, stroke_width=style.stroke, stroke_fill=stroke_fill)
            x += font.getlength(ch) + style.tracking * style.size

    if style.shadow:
        shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        a = int(255 * style.shadow_alpha)
        draw_text(ImageDraw.Draw(shadow), (0, 0, 0, a), (0, 0, 0, a))
        shadow = shadow.filter(ImageFilter.GaussianBlur(style.shadow))
        img = Image.alpha_composite(img, shadow)
        draw = ImageDraw.Draw(img)
    draw_text(draw, style.color, style.stroke_color)
    return np.asarray(img).copy()


@lru_cache(maxsize=64)
def render_emoji(char: str, size: int = 96) -> np.ndarray:
    """Render one colour emoji to RGBA. Segoe UI Emoji renders at size 109 natively; others scale."""
    font = load_font(EMOJI_FONT, 109)
    img = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((80, 80), char, font=font, anchor="mm", embedded_color=True)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    if img.width != size:
        s = size / max(img.width, img.height)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
    return np.asarray(img).copy()


def scale_rgba(rgba: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-3:
        return rgba
    img = Image.fromarray(rgba)
    w, h = max(1, int(img.width * factor)), max(1, int(img.height * factor))
    return np.asarray(img.resize((w, h), Image.BILINEAR)).copy()
