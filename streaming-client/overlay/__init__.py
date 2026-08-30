"""Overlays: per-frame decoration of the outgoing broadcast.

`base.py` is the contract; `status.py` is the shipped queue/now-playing
overlay. Overlays are chosen in code (`main.py`), not configuration —
`OVERLAY_ENABLED` only switches the shipped one on and off.
"""

from __future__ import annotations

from .base import Overlay
from .status import StreamStatusOverlay

__all__ = ["Overlay", "StreamStatusOverlay"]
