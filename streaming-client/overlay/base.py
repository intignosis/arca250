"""The overlay contract: draw live status onto every outgoing frame.

An `Overlay` sits inside the pacer's tick: after the pacer picks the frame to
send (a live model frame, a repeat, or black), it hands it to
`compose`, and what comes back is what the sink delivers. The overlay is
therefore on every frame of the broadcast — including the black holds between
clips, which is exactly where status information earns its keep.

Overlays are code, not configuration: the shipped one is `status.py`, and
changing what the broadcast shows means editing or replacing an
implementation of this class (wired in `main.py`). Only the on/off switch
lives in the environment (`OVERLAY_ENABLED`).

Rules for implementers — the pacer's clock is the constraint:

  * `compose` runs once per frame period (24 fps → ~41 ms budget shared with
    encoding). Stay well under a couple of milliseconds: pre-render text and
    panels into cached RGBA rasters when state changes, and per frame do
    nothing but numpy blends of those rasters.
  * **Never mutate the input frame.** The pacer re-sends its last frame
    while the model idles; drawing into it would bake the overlay in and
    accumulate. Return the input untouched when there is nothing to draw,
    else blend onto a copy.
  * Get state by listening (`ReactorLink.add_listener`) and reading the
    link's mirrors — never by sending commands or doing I/O on the compose
    path.
  * Raising is survivable (the pacer catches and keeps broadcasting) but a
    raise per tick floods the log; treat exceptions as bugs, not flow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Overlay(ABC):
    """One per-frame decoration pass over the outgoing broadcast."""

    @abstractmethod
    def compose(self, frame: np.ndarray) -> np.ndarray:
        """Return the frame to send: the input untouched, or a decorated copy.

        `frame` is the pacer's outgoing rgb24 array, shape (height, width, 3)
        uint8 — treat it as read-only.
        """
