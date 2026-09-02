"""The metadata group tag: this client's format for fast-h3 clip metadata.

The director writes it at enqueue time; the overlay and the director's own
narration read it back off the metadata echo. It lives in its own module
because both ends of the pipeline need it and neither should import the
other (the director sits upstream of the link, the overlay downstream of
the pacer — a shared import in either direction is a cycle).

The format itself is JSON with a `group_id` plus title, scene numbering,
author, source, `generated`, and the truncated raw prompt — see the
director's `_enqueue_group` for the authoritative writer.
"""

from __future__ import annotations

import json


def parse_group_tag(metadata: str) -> dict | None:
    """Read this client's group tag back out of a clip's metadata echo.

    Returns None for metadata this client did not write (other clients'
    clips, or an empty string).
    """
    try:
        tag = json.loads(metadata)
    except (TypeError, ValueError):
        return None
    if not isinstance(tag, dict) or "group_id" not in tag:
        return None
    return tag


def is_generated(clip: dict) -> bool:
    """Whether a clip is idle filler, judged purely from its metadata echo.

    Anything without a `generated: true` tag counts as viewer content —
    including untagged clips some other client enqueued.
    """
    tag = parse_group_tag(clip.get("metadata", ""))
    return bool(tag and tag.get("generated"))


def pick_next(clips: list[dict], ready_only: bool = True) -> dict | None:
    """The clip that should play next: viewer content first, then filler.

    The single playout policy, shared by the director (which sends the
    `play`) and the overlay (which shows "coming up") so the broadcast never
    announces one clip and plays another. Applied to the playout queue for
    the actual play decision; with `ready_only` false it ranks any list of
    clips (the overlay uses it on the generation queue to preview what will
    play once built). Within each class, queue order decides, so a group's
    scenes stay sequential.
    """
    pool = [c for c in clips if c.get("ready")] if ready_only else clips
    for clip in pool:
        if not is_generated(clip):
            return clip
    return pool[0] if pool else None


def viewer_insert_position(generation_clips: list[dict]) -> int | None:
    """Where a viewer clip enters the generation queue: ahead of filler.

    The index of the first filler clip — so viewer scenes land behind every
    viewer clip already waiting (first-come-first-served) and ahead of all
    idle filler, which just slides back. ``None`` when no filler waits:
    plain append is already the right spot.
    """
    for index, clip in enumerate(generation_clips):
        if is_generated(clip):
            return index
    return None
