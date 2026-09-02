"""The clip queues: what a session holds, in order, at each stage.

A clip passes three stages: enqueued (waiting in the **generation queue**),
built (waiting in the **playout queue**), and consumed (playing removed it).
Both queues are instances of one positional container, :class:`ClipQueue` —
bounded, ordered, with insert-at-position and move — and ``fasth3.py`` owns
when entries cross between them (a finished build leaves generation and joins
the playout tail).

Pure bookkeeping — no torch, no fastvideo, no runtime imports — so queue
behaviour is testable on any machine. `ClipInfo` in ``fasth3_types.py`` is
the wire form every mention of a clip carries; ``ClipEntry.snapshot()`` here
is its single producer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import fasth3_clip_plan as clip_plan


@dataclass
class ClipEntry:
    """One clip, from request to built payload.

    The client-facing fields are frozen at enqueue time: the prompt and
    metadata as the client sent them, and the frame count and seed as the
    session's conditions stood. ``video`` and ``audio`` are filled in when the
    build completes; ``ready`` is derived from their presence.
    """

    clip_id: str
    prompt: str
    metadata: str
    frames: int
    seed: int
    # Set while a build for this entry is in flight, so the scheduler never
    # submits the same entry twice.
    building: bool = False
    # The built payload: decoded RGB frames and the wire-ready waveform.
    video: list[Any] | None = None
    audio: Any = None

    @property
    def ready(self) -> bool:
        """Whether the clip is built and can be played."""
        return self.video is not None

    @property
    def seconds(self) -> float:
        """Exact playout length, derived from the frame count."""
        return clip_plan.seconds_for_frames(self.frames)

    def snapshot(self) -> dict[str, Any]:
        """The clip's wire form — the `ClipInfo` structure, as a plain mapping.

        Every message that references a clip carries this whole structure, so a
        client never has to join an id against an earlier message. A mapping
        rather than a dataclass instance, because the wire encoder accepts only
        JSON-representable values; ``ClipInfo`` in ``fasth3_types.py`` is the
        schema-side declaration of this exact shape.
        """
        return {
            "clip_id": self.clip_id,
            "prompt": self.prompt,
            "metadata": self.metadata,
            "frames": self.frames,
            "seconds": round(self.seconds, 3),
            "seed": self.seed,
            "ready": self.ready,
        }


def new_entry(*, prompt: str, metadata: str, frames: int, seed: int) -> ClipEntry:
    """Mint one entry with a fresh UUID; `enqueue` is its only caller."""
    return ClipEntry(
        clip_id=str(uuid.uuid4()),
        prompt=prompt,
        metadata=metadata,
        frames=frames,
        seed=seed,
    )


@dataclass
class ClipQueue:
    """A bounded, ordered, position-addressable queue of :class:`ClipEntry`.

    One container serves both stages: the generation queue (builds consume
    the front) and the playout queue (bare `play` and autoplay take the
    front). Positions are explicit — `add` takes one, `move` changes one —
    and nothing reorders behind the client's back. Capacity comes from the
    deployment config; for the playout queue every entry holds a fully built
    clip in host memory, so that bound is also the memory budget.
    """

    capacity: int
    _entries: list[ClipEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError(f"queue capacity must be positive, got {self.capacity}")

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, entry: ClipEntry) -> bool:
        return any(existing is entry for existing in self._entries)

    @property
    def full(self) -> bool:
        """Whether another add would exceed the capacity."""
        return len(self._entries) >= self.capacity

    def add(self, entry: ClipEntry, position: int | None = None) -> int:
        """Insert *entry* and return the index it landed at.

        ``position`` of ``None`` appends; anything else is clamped into
        ``0..len``, 0 being the front.

        Raises:
            ValueError: If the queue is already at capacity.
        """
        if self.full:
            raise ValueError(f"the queue is full ({self.capacity} clips)")
        index = (
            len(self._entries)
            if position is None
            else max(0, min(int(position), len(self._entries)))
        )
        self._entries.insert(index, entry)
        return index

    def move(self, entry: ClipEntry, position: int) -> int:
        """Reposition *entry* and return the index it landed at (clamped)."""
        if entry not in self:
            raise ValueError("the clip is not in this queue")
        self._entries = [existing for existing in self._entries if existing is not entry]
        index = max(0, min(int(position), len(self._entries)))
        self._entries.insert(index, entry)
        return index

    def get(self, clip_id: str) -> ClipEntry | None:
        """The entry with *clip_id*, or ``None`` when this queue lacks it."""
        for entry in self._entries:
            if entry.clip_id == clip_id:
                return entry
        return None

    def head(self) -> ClipEntry | None:
        """The front entry, or ``None`` when the queue is empty."""
        return self._entries[0] if self._entries else None

    def next_to_build(self) -> ClipEntry | None:
        """The front-most entry no build is running for."""
        for entry in self._entries:
            if not entry.building:
                return entry
        return None

    def remove(self, entry: ClipEntry) -> None:
        """Take *entry* out; consuming a clip (build done, played, popped)."""
        self._entries = [existing for existing in self._entries if existing is not entry]

    def clear(self) -> int:
        """Drop every entry, built payloads included, and return how many."""
        cleared = len(self._entries)
        self._entries = []
        return cleared

    def snapshot(self) -> list[dict[str, Any]]:
        """Every entry's wire form, front first — one queue of `queue_update`."""
        return [entry.snapshot() for entry in self._entries]


__all__ = ["ClipEntry", "ClipQueue", "new_entry"]
