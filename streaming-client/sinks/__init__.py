"""Output sinks. `make_sink` is the one place a sink name maps to a class.

Adding a destination (LiveKit, an SFU, a file recorder, ...):
  1. Implement `StreamSink` (see `base.py` for the contract) in a new module.
  2. Add a branch to `make_sink` and a value for `SINK` in `.env.example`.
  3. Document it in the README's sink table.
"""

from __future__ import annotations

from .base import AudioFormat, StreamSink, VideoFormat
from .noop import NoOpSink
from .rtmp import RtmpSink

__all__ = ["AudioFormat", "NoOpSink", "RtmpSink", "StreamSink", "VideoFormat", "make_sink"]


def make_sink(
    name: str,
    *,
    rtmp_url: str | None = None,
    rtmp_video_bitrate_k: int = 4500,
    music_path: str | None = None,
    music_volume: float = 0.35,
) -> StreamSink:
    """Build the sink named by config."""
    if name == "noop":
        return NoOpSink()
    if name == "rtmp":
        if not rtmp_url:
            raise ValueError("the rtmp sink needs an RTMP URL")
        return RtmpSink(
            rtmp_url,
            video_bitrate_k=rtmp_video_bitrate_k,
            music_path=music_path,
            music_volume=music_volume,
        )
    raise ValueError(f"unknown sink {name!r}")
