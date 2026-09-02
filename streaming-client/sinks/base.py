"""The output-sink contract: where the paced A/V stream goes.

A `StreamSink` receives a perfectly regular stream — the pacer calls
`send_video` exactly once per frame period with one rgb24 frame of the fixed
size, and `send_audio` once per frame period with exactly one period's worth
of int16 samples — regardless of what the model or the Reactor connection is
doing. Idle gaps have already been filled with repeated/black frames and
silence by the time a sink sees them.

That contract is what makes new sinks small: an implementation only encodes
and forwards; it never worries about gaps, bursts, reconnects, or frame
geometry. To add a destination (LiveKit, an SFU, a file, ...) implement this
class and register it in `sinks/__init__.py`.

Rules for implementers:
  * `send_video` / `send_audio` are called from the asyncio event loop and
    MUST NOT block. Anything that can stall (a pipe, a socket) goes behind an
    internal thread or queue.
  * A sink owns its own recovery. If its transport dies it should try to
    restart itself and report health through `alive`; the caller never
    restarts a sink mid-run.
  * `stop` must be idempotent and safe to call at any point.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VideoFormat:
    """Geometry and rate of the paced video stream."""

    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class AudioFormat:
    """Sample layout of the paced audio stream (int16 PCM)."""

    sample_rate: int
    channels: int


class StreamSink(ABC):
    """One destination for the paced audio/video stream."""

    @abstractmethod
    async def start(self, video: VideoFormat, audio: AudioFormat) -> None:
        """Open the transport. Called once, before the first frame."""

    @abstractmethod
    def send_video(self, frame: np.ndarray) -> None:
        """Accept one rgb24 frame, shape (height, width, 3) uint8, C-contiguous."""

    @abstractmethod
    def send_audio(self, samples: np.ndarray) -> None:
        """Accept one tick of int16 mono samples, shape (n,)."""

    @abstractmethod
    async def stop(self) -> None:
        """Close the transport and release resources. Idempotent."""

    @property
    @abstractmethod
    def alive(self) -> bool:
        """Whether the sink is currently able to deliver."""
