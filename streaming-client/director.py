"""The director: viewer prompts in, tagged scene groups on the model's queue.

One prompt from chat becomes one *scene group*: the upsampler expands it into
1..N self-contained scenes, and the director enqueues them back-to-back on the
fasth3 queue. Sequential playback needs no orchestration beyond that —
fasth3's queue is strict FIFO, builds run oldest-first, and autoplay plays the
oldest ready clip — so keeping a group's scenes contiguous *in the queue* is
what keeps them contiguous *on the stream*. Two rules protect that:

  * The director is the queue's only writer, and enqueues one group at a
    time, so groups can never interleave.
  * A group is only enqueued once the whole group fits in the remaining
    queue capacity, so it cannot get stuck half-in (with the model refusing
    the rest) while another group's turn comes up.

Every scene carries the group's identity in the clip's `metadata` — an opaque
string fasth3 stores and echoes back on every message that references the
clip. That is what lets this client (or any overlay built on it) reconstruct
"scene 2/3 of *Neon Alley* by viewer_42" from a `clip_started` alone, without
joining ids against local state that a reconnect may have lost.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from chat import ChatPrompt
from reactor_link import ReactorLink
from upsampler import PromptUpsampler, SceneGroup

logger = logging.getLogger(__name__)

# How many chat prompts may wait for upsampling+enqueue before new ones are
# turned away. Depth here is viewer wait time: at ~10s of video per scene the
# model plays through a full generation queue in a couple of minutes; a long
# backlog on top of that serves nobody.
_PENDING_LIMIT = 24

# Enqueue retry cadence while the model refuses (queue full, reconnect, ...).
_RETRY_DELAY_S = 3.0
_CAPACITY_POLL_S = 1.0


class Director:
    """Consume chat prompts; keep the fasth3 queue fed with scene groups."""

    def __init__(
        self,
        link: ReactorLink,
        upsampler: PromptUpsampler,
        cooldown_s: float,
    ) -> None:
        self._link = link
        self._upsampler = upsampler
        self._cooldown_s = cooldown_s
        self._pending: asyncio.Queue[ChatPrompt] = asyncio.Queue(_PENDING_LIMIT)
        self._last_accepted: dict[str, float] = {}  # author -> monotonic
        link.add_listener(self._on_model_message)

    # -------------------------------------------------------- chat intake

    def submit(self, prompt: ChatPrompt) -> None:
        """Accept one chat prompt (called synchronously by chat sources)."""
        now = time.monotonic()
        last = self._last_accepted.get(prompt.author)
        if last is not None and now - last < self._cooldown_s:
            logger.info(
                "[director] cooldown: dropping prompt from %s (%.0fs left)",
                prompt.author, self._cooldown_s - (now - last),
            )
            return
        try:
            self._pending.put_nowait(prompt)
        except asyncio.QueueFull:
            logger.warning(
                "[director] backlog full (%d); dropping prompt from %s",
                _PENDING_LIMIT, prompt.author,
            )
            return
        self._last_accepted[prompt.author] = now
        logger.info(
            "[director] accepted from %s@%s: %s",
            prompt.author, prompt.source, prompt.text,
        )

    # --------------------------------------------------------- main loop

    async def run(self) -> None:
        """Upsample and enqueue pending prompts, one group at a time, forever."""
        while True:
            prompt = await self._pending.get()
            try:
                group = await self._upsampler.upsample(
                    raw_prompt=prompt.text,
                    author=prompt.author,
                    source=prompt.source,
                    min_seconds=self._link.min_seconds,
                    max_seconds=self._link.max_seconds,
                )
                await self._enqueue_group(group)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error(
                    "[director] failed to process prompt from %s: %s",
                    prompt.author, error,
                )

    async def _enqueue_group(self, group: SceneGroup) -> None:
        scene_count = len(group.scenes)
        # Hold until the whole group fits, so its scenes land contiguously.
        while self._link.queue_capacity - self._link.queued < scene_count:
            await asyncio.sleep(_CAPACITY_POLL_S)

        for index, scene in enumerate(group.scenes, start=1):
            metadata = json.dumps(
                {
                    "group_id": group.group_id,
                    "title": group.title[:120],
                    "scene": index,
                    "scenes": scene_count,
                    "author": group.author,
                    "source": group.source,
                    # Truncated so the whole blob stays well under fasth3's
                    # 2000-char metadata cap.
                    "raw_prompt": group.raw_prompt[:400],
                },
                ensure_ascii=False,
            )
            while True:
                reply = await self._link.send_command(
                    "enqueue",
                    {
                        "prompt": scene.prompt,
                        "metadata": metadata,
                        "seconds": scene.seconds,
                    },
                )
                if isinstance(reply, dict) and "clip" in reply:
                    clip = reply["clip"]
                    logger.info(
                        "[director] queued %s scene %d/%d as %s (%.1fs, seed %s)",
                        group.group_id, index, scene_count,
                        clip["clip_id"][:8], clip["seconds"], clip["seed"],
                    )
                    break
                # Bodyless reply = refused (command_error was logged by the
                # link) or the session dropped mid-command; wait and retry.
                logger.warning(
                    "[director] enqueue of %s scene %d/%d refused; retrying in %.0fs",
                    group.group_id, index, scene_count, _RETRY_DELAY_S,
                )
                await asyncio.sleep(_RETRY_DELAY_S)

    # ----------------------------------------------------- announcements

    def _on_model_message(self, kind: str, data: dict) -> None:
        """Narrate group playback from clip messages alone (via metadata)."""
        clip = data.get("clip") if isinstance(data, dict) else None
        if not isinstance(clip, dict):
            return
        tag = _parse_group_tag(clip.get("metadata", ""))
        label = (
            f"'{tag['title']}' scene {tag['scene']}/{tag['scenes']} "
            f"(by {tag['author']}@{tag['source']})"
            if tag
            else f"clip {clip.get('clip_id', '?')[:8]}"
        )
        if kind == "clip_started":
            logger.info("[now playing] %s", label)
        elif kind == "clip_finished":
            logger.info("[finished] %s", label)
        elif kind == "clip_failed":
            logger.error(
                "[director] build failed for %s: %s — the queue moves on",
                label, data.get("reason"),
            )


def _parse_group_tag(metadata: str) -> dict | None:
    """Read this client's group tag back out of a clip's metadata echo."""
    try:
        tag = json.loads(metadata)
    except (TypeError, ValueError):
        return None
    if not isinstance(tag, dict) or "group_id" not in tag:
        return None
    return tag
