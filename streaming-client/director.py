"""The director: viewer prompts in, tagged scene groups on the model's queue.

One prompt from chat becomes one *scene group*: the upsampler expands it into
1..N self-contained scenes — a single shot, or a chunked short story — and
the director enqueues them contiguously on the fasth3 queue. The director is
also the *playout* brain: autoplay stays off, and `run_playout` sends an
explicit `play` for the next clip whenever the stream idles — viewer content
before filler, judged purely from the metadata echo (`pick_next` in
`group_tag.py`), because who asked for a clip is client-side knowledge the
model deliberately never has. Rules that keep it coherent:

  * The director is the queue's only writer. Both writers inside it — the
    viewer worker (`run`) and the idle filler (`run_idle`) — serialize group
    enqueues through one lock, so groups can never interleave.
  * A group is only enqueued when the whole group fits the remaining
    capacity (after any eviction), so it cannot get stuck half-in; a queue
    already full of viewer content drops new prompts instead of stalling
    them behind a wait. Capacity is whatever the connected deployment
    reports in `state_update`, never a constant.
  * Viewer prompts outrank filler, and stay first-come-first-served among
    themselves: an arriving viewer group pops every *unbuilt* filler clip
    (built ones survive as the stream's fallback) and appends — behind
    waiting viewer clips, ahead of nothing else unbuilt — then eviction
    (`pop`, newest first) reclaims built-filler slots when capacity needs
    it, and the filler stands down whenever viewer work is pending.

Every scene carries the group's identity in the clip's `metadata` — an opaque
string fasth3 stores and echoes back on every message that references the
clip. That is what lets this client (or any overlay built on it) reconstruct
"scene 2/3 of *Neon Alley* by viewer_42" from a `clip_started` alone, without
joining ids against local state that a reconnect may have lost. The same tag
carries `generated: true` on filler clips, which is what makes them
recognizably evictable later — including by a director restarted with no
memory of enqueueing them.

Viewer prompts pass moderation before they reach the upsampler; the curated
idle list does not need it (see `moderator.py` for the policy).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Sequence

from chat import ChatPrompt
from group_tag import is_generated, parse_group_tag, pick_next
from moderator import Moderator
from reactor_link import ReactorLink
from upsampler import PromptUpsampler, SceneGroup

logger = logging.getLogger(__name__)

# How many chat prompts may wait for upsampling+enqueue before new ones are
# turned away. Depth here is viewer wait time: at ~10s of video per scene the
# model plays through a full generation queue in a couple of minutes; a long
# backlog on top of that serves nobody.
_PENDING_LIMIT = 24

# Enqueue retry cadence while the model refuses (reconnect mid-command, ...).
_RETRY_DELAY_S = 3.0

# How often the idle filler re-checks whether the queue wants topping up.
_IDLE_POLL_S = 3.0

# How often the playout loop re-checks for an idle stream with a ready clip.
# state_update/queue_update broadcasts keep the mirrors fresh; polling them is
# what survives a missed message.
_PLAYOUT_POLL_S = 0.5


class Director:
    """Consume chat prompts; keep the fasth3 queue fed with scene groups."""

    def __init__(
        self,
        link: ReactorLink,
        upsampler: PromptUpsampler,
        moderator: Moderator,
        cooldown_s: float,
        idle_prompts: Sequence[str] = (),
        idle_queue_target: int = 0,
    ) -> None:
        self._link = link
        self._upsampler = upsampler
        self._moderator = moderator
        self._cooldown_s = cooldown_s
        self._idle_prompts = list(idle_prompts)
        random.shuffle(self._idle_prompts)
        self._idle_index = 0
        self._idle_target = idle_queue_target
        self._pending: asyncio.Queue[ChatPrompt] = asyncio.Queue(_PENDING_LIMIT)
        self._last_accepted: dict[str, float] = {}  # author -> monotonic
        self._enqueue_lock = asyncio.Lock()
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

    # ------------------------------------------------- viewer prompt loop

    def _viewer_clips_queued(self) -> int:
        """Queued clips that are viewer content (anything not tagged filler)."""
        return sum(
            1 for clip in self._link.queue_clips if not is_generated(clip)
        )

    async def run(self) -> None:
        """Moderate, upsample, and enqueue pending prompts, one group at a time."""
        while True:
            prompt = await self._pending.get()
            try:
                # A queue already full of viewer content takes no more: the
                # prompt is dropped now, before it costs a moderation and an
                # LLM call. Capacity is whatever the connected deployment
                # reports, never a constant.
                if self._viewer_clips_queued() >= self._link.queue_capacity:
                    logger.warning(
                        "[director] queue is full of viewer clips (%d/%d); "
                        "dropping prompt from %s",
                        self._viewer_clips_queued(), self._link.queue_capacity,
                        prompt.author,
                    )
                    continue
                verdict = await self._moderator.review(prompt.text)
                if verdict is not None:
                    logger.warning(
                        "[director] rejected prompt from %s@%s (%s): %s",
                        prompt.author, prompt.source, verdict, prompt.text,
                    )
                    continue
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

    # -------------------------------------------------------- idle filler

    async def run_idle(self) -> None:
        """Keep the queue topped up with generated clips while chat is quiet.

        One clip per group, on purpose: single-scene fillers are the finest
        eviction granularity, and popping one never truncates a story.
        """
        if not self._idle_prompts or self._idle_target <= 0:
            logger.info("[director] idle filler disabled (no prompts or target 0)")
            return
        logger.info(
            "[director] idle filler: %d prompts, queue target %d",
            len(self._idle_prompts), self._idle_target,
        )
        while True:
            await asyncio.sleep(_IDLE_POLL_S)
            if (
                not self._pending.empty()
                or not self._link.connected
                or self._link.queued >= self._idle_target
            ):
                continue
            text = self._idle_prompts[self._idle_index % len(self._idle_prompts)]
            self._idle_index += 1
            try:
                group = await self._upsampler.upsample(
                    raw_prompt=text,
                    author="auto",
                    source="idle",
                    min_seconds=self._link.min_seconds,
                    max_seconds=self._link.max_seconds,
                    generated=True,
                    max_chunks=1,
                )
                # A viewer prompt that arrived while the LLM ran outranks the
                # filler; drop this group rather than making the viewer wait.
                if self._pending.empty():
                    await self._enqueue_group(group)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error("[director] idle fill failed: %s", error)

    # ------------------------------------------------------------- playout

    async def run_playout(self) -> None:
        """Play the next clip whenever the stream is idle and one is ready.

        Autoplay is deliberately off: the model plays oldest-first, but this
        stream wants viewer content before filler, and who asked for a clip
        is client-side knowledge (the metadata echo). `pick_next` is the
        policy — first ready non-generated clip in queue order, else the
        oldest ready one — and the overlay's "coming up" uses the same
        function, so what is announced is what plays.
        """
        while True:
            await asyncio.sleep(_PLAYOUT_POLL_S)
            if not self._link.connected or self._link.state.get("playing"):
                continue
            choice = pick_next(self._link.queue_clips)
            if choice is None:
                continue
            await self._link.send_command("play", {"clip_id": choice["clip_id"]})
            # Let the resulting state_update land before re-checking; a race
            # (the clip started on its own accord, or was popped) surfaces as
            # a refusal the link logs, and the next tick re-evaluates.
            await asyncio.sleep(_PLAYOUT_POLL_S)

    # ---------------------------------------------------------- enqueueing

    async def _enqueue_group(self, group: SceneGroup) -> None:
        """Put one group on the model's queue, or drop it and say why.

        Viewer groups take precedence over filler in build order the direct
        way: every *unbuilt* filler clip is popped first (`run_idle` refills
        once chat goes quiet), so the appended viewer scenes are the front of
        the unbuilt segment — behind any viewer clips already waiting, which
        keeps viewer requests first-come-first-served. Built filler survives
        for the stream to fall back on; the playout policy already prefers
        viewer clips over it. When even evicting every filler cannot fit the
        group, the group is dropped — a queue full of viewer content takes
        no more, rather than stalling every later prompt behind a wait.
        """
        scene_count = len(group.scenes)
        async with self._enqueue_lock:
            free = self._link.queue_capacity - self._link.queued
            if not group.generated:
                # Judge the fit before touching anything: when even evicting
                # every filler could not make room, drop the group with the
                # queue intact instead of spending fillers on a lost cause.
                evictable = sum(
                    1 for clip in self._link.queue_clips if is_generated(clip)
                )
                if free + evictable < scene_count:
                    logger.warning(
                        "[director] no room for %s (%d scenes, %d free, %d "
                        "evictable); dropping the group",
                        group.group_id, scene_count, free, evictable,
                    )
                    return
                await self._pop_unbuilt_fillers()
                free = self._link.queue_capacity - self._link.queued
                if free < scene_count:
                    popped = await self._evict_fillers(scene_count - free)
                    if popped:
                        # Give the pops' queue_update a moment to land.
                        await asyncio.sleep(0.3)
                    free = self._link.queue_capacity - self._link.queued
            if free < scene_count:
                logger.warning(
                    "[director] no room for %s (%d scenes, %d free, nothing "
                    "evictable); dropping the group",
                    group.group_id, scene_count, free,
                )
                return

            for index, scene in enumerate(group.scenes, start=1):
                metadata = json.dumps(
                    {
                        "group_id": group.group_id,
                        "title": group.title[:120],
                        "scene": index,
                        "scenes": scene_count,
                        "author": group.author,
                        "source": group.source,
                        "generated": group.generated,
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
                            "[director] queued %s scene %d/%d as %s (%.1fs, seed %s)%s",
                            group.group_id, index, scene_count,
                            clip["clip_id"][:8], clip["seconds"], clip["seed"],
                            " [auto]" if group.generated else "",
                        )
                        break
                    # Bodyless reply = refused (command_error was logged by the
                    # link) or the session dropped mid-command; wait and retry.
                    logger.warning(
                        "[director] enqueue of %s scene %d/%d refused; retrying in %.0fs",
                        group.group_id, index, scene_count, _RETRY_DELAY_S,
                    )
                    await asyncio.sleep(_RETRY_DELAY_S)

    async def _pop_unbuilt_fillers(self) -> int:
        """Pop every filler clip whose build has not finished.

        This is what puts an arriving viewer group at the head of the unbuilt
        queue without any model-side priority: with no unbuilt filler left,
        the appended viewer scenes are the next builds. A filler mid-build
        looks identical to a waiting one on the wire and gets popped too —
        the model discards its result on completion; the slot frees now.
        Returns how many pops succeeded.
        """
        popped = 0
        for clip in list(self._link.queue_clips):
            if clip.get("ready") or not is_generated(clip):
                continue
            reply = await self._link.send_command(
                "pop", {"clip_id": clip["clip_id"]}
            )
            if isinstance(reply, dict) and "clip" in reply:
                popped += 1
        if popped:
            logger.info(
                "[director] popped %d unbuilt filler clip(s) for a viewer group",
                popped,
            )
            await asyncio.sleep(0.3)
        return popped

    async def _evict_fillers(self, needed: int) -> int:
        """Pop up to `needed` generated clips to make room for a viewer group.

        Newest-queued first, so the filler closest to playing (and likeliest
        already built) survives and the stream stays fed. Only clips tagged
        `generated: true` are candidates; the playing clip is not in the
        queue, so it can never be popped. Returns how many pops succeeded.
        """
        popped = 0
        for clip in reversed(self._link.queue_clips):
            if popped >= needed:
                break
            tag = parse_group_tag(clip.get("metadata", ""))
            if not tag or not tag.get("generated"):
                continue
            reply = await self._link.send_command(
                "pop", {"clip_id": clip["clip_id"]}
            )
            if isinstance(reply, dict) and "clip" in reply:
                popped += 1
                logger.info(
                    "[director] evicted filler %s ('%s') for a viewer group",
                    clip["clip_id"][:8], tag.get("title", "?"),
                )
        return popped

    # ----------------------------------------------------- announcements

    def _on_model_message(self, kind: str, data: dict) -> None:
        """Narrate group playback from clip messages alone (via metadata)."""
        clip = data.get("clip") if isinstance(data, dict) else None
        if not isinstance(clip, dict):
            return
        tag = parse_group_tag(clip.get("metadata", ""))
        label = (
            f"'{tag['title']}' scene {tag['scene']}/{tag['scenes']} "
            f"(by {tag['author']}@{tag['source']})"
            + (" [auto]" if tag.get("generated") else "")
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
