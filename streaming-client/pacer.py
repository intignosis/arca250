"""The pacer: turn clip-shaped model output into a constant-rate broadcast.

fasth3's output is clips with black holds in between — frames arrive at a
strict 24 fps *while a clip plays* and not at all while the queue idles or the
Reactor connection is being rebuilt. An RTMP ingest (and any live sink) needs
the opposite: a frame every period and audio every period, forever, or players
stall and the platform drops the broadcast.

The pacer is the adapter between the two. It is a drift-free metronome at the
model's frame rate; each tick it:

  * pops the oldest buffered video frame (or repeats the last one shown, or
    black before anything arrived) and hands it to the sink;
  * pulls exactly one tick's worth of int16 samples from the audio buffer
    (padding with silence on underflow) and hands those to the sink.

Both media types are buffered FIFO with the same shallow cap, which is what
keeps them in sync: while a clip plays, both buffers stay near-empty and
frames flow through with the same tiny delay; while nothing plays, both run
dry and the pacer emits repeats + silence. Overflow (the model briefly ahead
of the clock) drops the oldest entries of each, and is counted.

The pacer never touches the Reactor connection and never stops on its own —
it is created once, outlives session reconnects, and is cancelled only at
shutdown. That is what keeps the platform-side stream uninterrupted while the
client rebuilds a session behind it.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time

import numpy as np

from overlay import Overlay
from sinks import AudioFormat, StreamSink, VideoFormat

logger = logging.getLogger(__name__)

# How much media may sit between the model and the sink before the oldest is
# dropped. Shallow on purpose: depth here is end-to-end latency.
_BUFFER_SECONDS = 2.0

# If the loop is starved long enough to fall this many periods behind, resnap
# the clock instead of machine-gunning catch-up frames into the sink.
_RESNAP_PERIODS = 8


class Pacer:
    """Constant-rate A/V clock between the model callbacks and one sink."""

    def __init__(
        self,
        sink: StreamSink,
        video: VideoFormat,
        audio: AudioFormat,
        overlay: Overlay | None = None,
    ) -> None:
        if audio.sample_rate % video.fps != 0:
            raise ValueError(
                f"sample rate {audio.sample_rate} must divide evenly by fps {video.fps}"
            )
        self._sink = sink
        self._video = video
        self._audio = audio
        self._overlay = overlay
        self._overlay_errors = 0
        self._samples_per_tick = audio.sample_rate // video.fps

        max_frames = int(video.fps * _BUFFER_SECONDS)
        self._frames: collections.deque[np.ndarray] = collections.deque(maxlen=max_frames)
        self._audio_chunks: collections.deque[np.ndarray] = collections.deque()
        self._audio_buffered = 0  # samples across _audio_chunks
        self._max_audio_samples = int(audio.sample_rate * _BUFFER_SECONDS)

        self._black = np.zeros((video.height, video.width, 3), dtype=np.uint8)
        self._silence = np.zeros(self._samples_per_tick, dtype=np.int16)
        self._last_frame = self._black

        # Counters, logged periodically and readable by anyone.
        self.ticks = 0
        self.repeated_frames = 0
        self.silent_ticks = 0
        self.dropped_frames = 0
        self.dropped_samples = 0

    # ------------------------------------------------- model-facing intake

    def submit_video(self, frame: np.ndarray) -> None:
        """Buffer one model frame. Called from the track's frame callback."""
        frame = np.asarray(frame)
        if frame.shape[:2] != (self._video.height, self._video.width):
            frame = self._fit(frame)
        if len(self._frames) == self._frames.maxlen:
            self.dropped_frames += 1
        self._frames.append(frame)

    def submit_audio(self, samples: np.ndarray) -> None:
        """Buffer model audio (int16, any chunk size; channels are flattened)."""
        flat = np.asarray(samples, dtype=np.int16).reshape(-1)
        if flat.size == 0:
            return
        self._audio_chunks.append(flat)
        self._audio_buffered += flat.size
        while self._audio_buffered > self._max_audio_samples:
            oldest = self._audio_chunks.popleft()
            self._audio_buffered -= oldest.size
            self.dropped_samples += oldest.size

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        """Center a differently-sized frame on the fixed black canvas.

        The canvas is fixed for the sink's lifetime (raw-video geometry cannot
        change mid-stream), so a model frame of another size — e.g. after a
        `set_canvas` this client never sends — is letterboxed, not resized;
        no interpolation dependency, and it cannot garble the stream.
        """
        height, width = self._video.height, self._video.width
        crop = frame[:height, :width, :3]
        canvas = self._black.copy()
        top = (height - crop.shape[0]) // 2
        left = (width - crop.shape[1]) // 2
        canvas[top : top + crop.shape[0], left : left + crop.shape[1]] = crop
        return canvas

    def _pull_audio_tick(self) -> np.ndarray:
        """Exactly one tick of samples: buffered audio padded with silence."""
        needed = self._samples_per_tick
        if self._audio_buffered == 0:
            self.silent_ticks += 1
            return self._silence
        parts: list[np.ndarray] = []
        while needed > 0 and self._audio_chunks:
            chunk = self._audio_chunks[0]
            if chunk.size <= needed:
                parts.append(self._audio_chunks.popleft())
                needed -= chunk.size
            else:
                parts.append(chunk[:needed])
                self._audio_chunks[0] = chunk[needed:]
                needed = 0
        pulled = np.concatenate(parts) if len(parts) > 1 else parts[0]
        self._audio_buffered -= pulled.size
        if needed > 0:
            pulled = np.concatenate([pulled, np.zeros(needed, dtype=np.int16)])
        return pulled

    # ------------------------------------------------------------ the clock

    async def run(self) -> None:
        """Tick forever at the frame rate; cancelled only at shutdown."""
        await self._sink.start(self._video, self._audio)
        period = 1.0 / self._video.fps
        next_tick = time.monotonic() + period
        last_report = time.monotonic()

        while True:
            delay = next_tick - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            elif -delay > period * _RESNAP_PERIODS:
                logger.warning(
                    "[pacer] %.2fs behind schedule; resnapping the clock", -delay
                )
                next_tick = time.monotonic()
            next_tick += period

            if self._frames:
                self._last_frame = self._frames.popleft()
            else:
                self.repeated_frames += 1
            outgoing = self._last_frame
            if self._overlay is not None:
                # An overlay bug must not take the broadcast down; compose
                # never mutates _last_frame, so the clean frame survives.
                try:
                    outgoing = self._overlay.compose(outgoing)
                except Exception:
                    self._overlay_errors += 1
                    if self._overlay_errors in (1, 100) or self._overlay_errors % 10_000 == 0:
                        logger.exception(
                            "[pacer] overlay compose failed (%d times)",
                            self._overlay_errors,
                        )
            self._sink.send_video(outgoing)
            self._sink.send_audio(self._pull_audio_tick())
            self.ticks += 1

            now = time.monotonic()
            if now - last_report >= 60.0:
                logger.info(
                    "[pacer] ticks=%d live_frames=%d repeats=%d "
                    "silent_ticks=%d dropped=%df/%.1fs-audio",
                    self.ticks,
                    self.ticks - self.repeated_frames,
                    self.repeated_frames,
                    self.silent_ticks,
                    self.dropped_frames,
                    self.dropped_samples / self._audio.sample_rate,
                )
                last_report = now
