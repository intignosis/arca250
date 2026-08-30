"""Which commands the session would accept right now.

Kept out of ``fasth3.py`` so the state machine is readable on its own and can be
tested without a GPU. ``StateUpdate.valid_commands`` carries the result to
clients, so a frontend enables and greys out controls from the snapshot instead
of re-deriving these rules.
"""

from __future__ import annotations

# Always available: they only ever record a value or report one.
_ALWAYS = ("set_clip_seconds", "set_seed", "set_autoplay", "get_queue", "get_state", "reset")


def valid_commands(
    *,
    playing: bool,
    generation_queued: int,
    generation_capacity: int,
    playout_queued: int,
) -> list[str]:
    """Name every command the session would accept in this state.

    Args:
        playing: A clip is streaming on the output tracks.
        generation_queued: Clips waiting in the generation queue.
        generation_capacity: Most clips the generation queue holds.
        playout_queued: Built clips waiting in the playout queue.
    """
    commands = list(_ALWAYS)
    if generation_queued < generation_capacity:
        commands.append("enqueue")
    if generation_queued > 0 or playout_queued > 0:
        commands.append("pop")
        commands.append("move")
    if playing:
        commands.append("stop")
    else:
        if playout_queued > 0:
            commands.append("play")
        # The canvas fixes the video track's geometry and the shape queued
        # clips were built at, so it can only change while everything is
        # empty.
        if generation_queued == 0 and playout_queued == 0:
            commands.append("set_canvas")
    return sorted(commands)


__all__ = ["valid_commands"]
