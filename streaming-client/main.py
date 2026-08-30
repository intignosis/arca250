"""Chat-driven fasth3 livestream client. See README.md for the full picture.

Wiring, in dependency order:

  chat sources ──▶ Director ──▶ PromptUpsampler (OpenAI-compatible LLM)
                      │
                      ▼ enqueue (scene groups, tagged via metadata) + play
                 ReactorLink ◀──▶ served fasth3 model (local or hosted)
                      │ frames / audio
                      ▼
                    Pacer ──▶ StreamSink (rtmp | noop)

Everything is one asyncio process. The pacer and sink are created after the
first `state_update` (that is where the deployment's canvas size comes from)
and then live until shutdown, across any number of Reactor reconnects.

Usage:
    cp .env.example .env      # fill in keys, style, sink, chat
    python main.py            # everything from .env
    python main.py --local --sink noop     # local runtime, throwaway output
    python main.py --sink rtmp --rtmp-url rtmp://live.twitch.tv/app/KEY
"""

from __future__ import annotations

import asyncio
import logging
import warnings

from chat import ChatSource, TwitchChat, YouTubeChat
from config import Config
from director import Director
from moderator import Moderator
from overlay import StreamStatusOverlay
from pacer import Pacer
from reactor_link import MODEL_FPS, MODEL_SAMPLE_RATE, ReactorLink
from sinks import AudioFormat, VideoFormat, make_sink
from upsampler import PromptUpsampler

logger = logging.getLogger("streaming-client")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    # WebRTC internals are chatty at INFO and alarming at their defaults.
    logging.getLogger("aiortc.codecs.vpx").setLevel(logging.ERROR)
    logging.getLogger("aiortc.codecs.h264").setLevel(logging.ERROR)
    logging.getLogger("aioice.ice").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def build_chat_sources(config: Config) -> list[ChatSource]:
    """One source per configured platform. Add new platforms here."""
    sources: list[ChatSource] = []
    if config.twitch_channel:
        sources.append(TwitchChat(config.twitch_channel, config.chat_command))
    if config.youtube_video_id and config.youtube_api_key:
        sources.append(
            YouTubeChat(
                config.youtube_video_id, config.youtube_api_key, config.chat_command
            )
        )
    return sources


async def main() -> None:
    setup_logging()
    config = Config.load()

    link = ReactorLink(config)
    upsampler = PromptUpsampler(
        api_key=config.openai_api_key,
        model=config.openai_model,
        style=config.style,
        max_chunks=config.max_chunks,
        base_url=config.openai_base_url,
    )
    moderator = Moderator(
        api_key=config.moderation_api_key,
        model=config.moderation_model,
        enabled=config.moderation_enabled,
        base_url=config.moderation_base_url,
    )
    if not moderator.enabled:
        logger.warning(
            "moderation is DISABLED (MODERATION_ENABLED=0) — every chat "
            "prompt reaches the upsampler unchecked"
        )
    director = Director(
        link,
        upsampler,
        moderator,
        cooldown_s=config.chat_cooldown_s,
        idle_prompts=config.idle_prompts,
        idle_queue_target=config.idle_queue_target,
    )
    chat_sources = build_chat_sources(config)
    if not chat_sources:
        logger.warning(
            "no chat source configured (TWITCH_CHANNEL / YOUTUBE_VIDEO_ID) — "
            "the stream will run, but nothing will feed the queue"
        )
    sink = make_sink(
        config.sink,
        rtmp_url=config.rtmp_url,
        rtmp_video_bitrate_k=config.rtmp_video_bitrate_k,
    )

    tasks: list[asyncio.Task] = [
        asyncio.create_task(link.run(), name="reactor-link"),
        asyncio.create_task(director.run(), name="director"),
        asyncio.create_task(director.run_playout(), name="playout"),
    ]
    # Gated here because main treats any finished task as a shutdown signal,
    # and run_idle returns immediately when the filler is configured off.
    if config.idle_prompts and config.idle_queue_target > 0:
        tasks.append(asyncio.create_task(director.run_idle(), name="idle-filler"))
    else:
        logger.info("idle filler off (no prompts file or IDLE_QUEUE_TARGET=0)")
    tasks += [
        asyncio.create_task(source.run(director.submit), name=f"chat-{source.name}")
        for source in chat_sources
    ]

    try:
        # The sink's geometry comes from the deployment (state_update), so the
        # pacer starts only once the first session is up. From then on it and
        # the sink survive every reconnect.
        await link.wait_first_state()
        width, height = link.canvas
        # The overlay to broadcast is a code decision; swap the class here.
        overlay = (
            StreamStatusOverlay(link, chat_command=config.chat_command)
            if config.overlay_enabled
            else None
        )
        pacer = Pacer(
            sink,
            VideoFormat(width=width, height=height, fps=MODEL_FPS),
            AudioFormat(sample_rate=MODEL_SAMPLE_RATE, channels=1),
            overlay=overlay,
        )
        link.attach_pacer(pacer)
        tasks.append(asyncio.create_task(pacer.run(), name="pacer"))
        logger.info(
            "streaming %dx%d@%dfps to sink=%s (overlay %s) — chat command %r on %s",
            width, height, MODEL_FPS, config.sink,
            "on" if overlay else "off",
            config.chat_command,
            ", ".join(s.name for s in chat_sources) or "nothing",
        )

        # Run until a task dies (none should) or the process is interrupted.
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task.exception() is not None:
                logger.error("task %s died: %s", task.get_name(), task.exception())
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for source in chat_sources:
            try:
                await source.close()
            except Exception:
                pass
        await sink.stop()
        logger.info("shut down cleanly")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
