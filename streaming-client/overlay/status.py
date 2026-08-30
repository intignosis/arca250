"""The shipped overlay: queue depth, what is playing, and what comes next.

Layout, tuned to stay out of the picture (small type, thin translucent
plates, 16 px margins on a 1344×768 canvas):

  top-left    NOW  <title> — scene 2/3 · by <author>     (while playing)
              COMING UP  <next title> · by <author>       (dimmer, beneath)
  top-left    UP NEXT  <title> · by <author>              (while idle)
  top-left    type "!prompt <your idea>" in chat          (idle, queue empty)
  top-right   QUEUE 4/10

All content comes from the metadata group tags the director writes at
enqueue time (title, author, scene numbering, `generated`) and from the
link's `state_update`/`queue_update` mirrors — so the overlay reconstructs
everything from the wire, survives client restarts, and shows `auto` as the
author of idle-filler clips. A clip enqueued by some other client (no tag)
degrades to its prompt text.

Rendering: panels are rasterized with Pillow only when their text changes,
cached as RGBA arrays, and alpha-blended onto a copy of the frame with plain
numpy per tick — the compose path does no text layout.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from typing import TYPE_CHECKING

from group_tag import parse_group_tag, pick_next

from .base import Overlay

if TYPE_CHECKING:
    from reactor_link import ReactorLink

logger = logging.getLogger(__name__)

_MARGIN = 16
_LINE_GAP = 6
_MAX_TEXT = 72  # characters per line, before ellipsis

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)

# (text alpha, plate alpha) per style; the dim style is the "coming up" line.
_STYLES = {
    "primary": (235, 150),
    "dim": (160, 95),
    "badge": (225, 150),
}


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    logger.warning("[overlay] no TrueType font found; using PIL's bitmap font")
    return ImageFont.load_default()


class StreamStatusOverlay(Overlay):
    """Draw queue depth, the playing scene, and the next one, per frame."""

    def __init__(self, link: "ReactorLink", chat_command: str = "!prompt") -> None:
        self._link = link
        self._chat_command = chat_command
        self._current: dict | None = None  # display info of the playing clip
        self._font = _load_font(19)
        self._font_small = _load_font(16)
        link.add_listener(self._on_message)

    # -------------------------------------------------------------- state

    def _on_message(self, kind: str, data: dict) -> None:
        if kind == "clip_started":
            self._current = _clip_display(data.get("clip"))
        elif kind in ("clip_finished", "clip_stopped"):
            self._current = None

    def _up_next(self) -> dict | None:
        """Display info for the clip the director will play next.

        The same `pick_next` the playout loop uses (viewer content before
        filler, ready clips preferred), so the overlay never announces one
        clip while another plays. With nothing ready yet, the preference
        runs over the whole queue: what will play once built.
        """
        clips = self._link.queue_clips
        choice = pick_next(clips) or pick_next(clips, ready_only=False)
        return _clip_display(choice)

    # --------------------------------------------------------- composing

    def compose(self, frame: np.ndarray) -> np.ndarray:
        current, up_next = self._current, self._up_next()

        if current is not None:
            primary = f"NOW  {_headline(current, with_scene=True)}"
            secondary = None
            if up_next is not None:
                if up_next["group_id"] and up_next["group_id"] == current["group_id"]:
                    secondary = f"COMING UP  scene {up_next['scene']}/{up_next['scenes']}"
                else:
                    secondary = f"COMING UP  {_headline(up_next)}"
        elif up_next is not None:
            primary = f"UP NEXT  {_headline(up_next)}"
            secondary = None
        else:
            primary = f'type "{self._chat_command} <your idea>" in chat'
            secondary = None

        badge = f"QUEUE {self._link.queued}/{self._link.queue_capacity}"

        out = frame.copy()
        x = y = _MARGIN
        panel = _panel(primary, self._font, "primary")
        _blend(out, panel, x, y)
        if secondary:
            y += panel.shape[0] + _LINE_GAP
            _blend(out, _panel(secondary, self._font_small, "dim"), x, y)

        badge_panel = _panel(badge, self._font_small, "badge")
        _blend(out, badge_panel, out.shape[1] - _MARGIN - badge_panel.shape[1], _MARGIN)
        return out


def _clip_display(clip: dict | None) -> dict | None:
    """Reduce a ClipInfo to what the overlay shows, tag or no tag."""
    if not isinstance(clip, dict):
        return None
    tag = parse_group_tag(clip.get("metadata", ""))
    if tag:
        return {
            "group_id": tag.get("group_id", ""),
            "title": str(tag.get("title") or "untitled"),
            "author": str(tag.get("author") or "?"),
            "scene": int(tag.get("scene") or 1),
            "scenes": int(tag.get("scenes") or 1),
        }
    return {
        "group_id": "",
        "title": str(clip.get("prompt", ""))[:60] or "untitled",
        "author": "",
        "scene": 1,
        "scenes": 1,
    }


def _headline(info: dict, with_scene: bool = False) -> str:
    text = _shorten(info["title"])
    if with_scene and info["scenes"] > 1:
        text += f" — scene {info['scene']}/{info['scenes']}"
    if info["author"]:
        text += f" · by {info['author']}"
    return text


def _shorten(text: str, limit: int = _MAX_TEXT) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# Raster cache keyed by rendered text + font + style. Bounded by wholesale
# clearing: state text changes a few times a minute, so 128 entries is weeks
# of variety, and a clear costs one re-render per visible panel.
_panel_store: dict[tuple[str, int, str], np.ndarray] = {}


def _panel(text: str, font, style: str) -> np.ndarray:
    """One translucent plate with `text` on it, as a cached (h, w, 4) raster."""
    key = (text, id(font), style)
    cached = _panel_store.get(key)
    if cached is not None:
        return cached
    if len(_panel_store) > 128:
        _panel_store.clear()

    text_alpha, plate_alpha = _STYLES[style]
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
    pad_x, pad_y = 10, 6
    width = right - left + 2 * pad_x
    height = bottom - top + 2 * pad_y

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (0, 0, width - 1, height - 1), radius=7, fill=(10, 10, 14, plate_alpha)
    )
    draw.text(
        (pad_x - left, pad_y - top), text, font=font,
        fill=(255, 255, 255, text_alpha),
    )
    raster = np.asarray(image, dtype=np.uint8)
    _panel_store[key] = raster
    return raster


def _blend(dst: np.ndarray, panel: np.ndarray, x: int, y: int) -> None:
    """Alpha-composite `panel` onto `dst` in place at (x, y), clipped."""
    height, width = panel.shape[:2]
    x, y = max(0, x), max(0, y)
    width = min(width, dst.shape[1] - x)
    height = min(height, dst.shape[0] - y)
    if width <= 0 or height <= 0:
        return
    region = dst[y : y + height, x : x + width]
    fg = panel[:height, :width]
    alpha = fg[:, :, 3:4].astype(np.uint16)
    region[:] = (
        (fg[:, :, :3].astype(np.uint16) * alpha + region.astype(np.uint16) * (255 - alpha))
        // 255
    ).astype(np.uint8)
