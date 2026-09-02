"""A sink that throws the stream away.

Useful for driving the whole pipeline — chat, upsampling, the queue, playback,
the pacer — without an RTMP endpoint: everything runs for real except delivery.
It counts what it discards and logs a heartbeat so you can see the pacer is
feeding it at the right rate.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .base import AudioFormat, StreamSink, VideoFormat

logger = logging.getLogger(__name__)

_HEARTBEAT_S = 10.0


class NoOpSink(StreamSink):
    """Discard every frame and sample; log a periodic heartbeat."""

    def __init__(self) -> None:
        self._frames = 0
        self._samples = 0
        self._started_at = 0.0
        self._last_beat = 0.0
        self._sample_rate = 48_000

    async def start(self, video: VideoFormat, audio: AudioFormat) -> None:
        self._sample_rate = audio.sample_rate
        self._started_at = self._last_beat = time.monotonic()
        logger.info(
            "[noop] started: %dx%d@%dfps, %dHz x%d audio — discarding everything",
            video.width, video.height, video.fps, audio.sample_rate, audio.channels,
        )

    def send_video(self, frame: np.ndarray) -> None:
        self._frames += 1
        now = time.monotonic()
        if now - self._last_beat >= _HEARTBEAT_S:
            elapsed = now - self._started_at
            logger.info(
                "[noop] %d frames / %.1fs audio discarded (%.2f fps average)",
                self._frames, self._samples / self._sample_rate, self._frames / elapsed,
            )
            self._last_beat = now

    def send_audio(self, samples: np.ndarray) -> None:
        self._samples += len(samples)

    async def stop(self) -> None:
        logger.info("[noop] stopped after %d frames", self._frames)

    @property
    def alive(self) -> bool:
        return True
