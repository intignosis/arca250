"""The chat-source contract: where `!prompt` requests come from.

A `ChatSource` watches one chat (a Twitch channel, a YouTube live chat, ...)
and calls the provided callback with a `ChatPrompt` for every message that
starts with one of the configured command words — the viewer prompt command
plus any admin commands (`admin.py`); the router in `main.py` tells them
apart by the `command` field. Everything else about the platform —
transport, polling cadence, reconnects — is the source's own business.

Rules for implementers:
  * `run` is a long-lived coroutine: connect, deliver prompts, and recover
    from transient failures internally (with backoff). Return only when
    cancelled or the source is permanently unusable (log why).
  * Deliver each message at most once, and nothing from before the source
    started — replaying a chat backlog at startup floods the queue.
  * Strip the command word before delivering; `text` is the bare payload,
    and `command` records which word matched.
  * The callback is synchronous, cheap, and never raises; call it inline.

To add a platform (Kick, Discord, ...): implement this class in a new module,
wire it in `main.py`'s `build_chat_sources`, and document the env vars in
`.env.example` and the README.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChatPrompt:
    """One accepted command message from a chatter, command word stripped."""

    source: str  # e.g. "twitch", "youtube"
    author: str
    text: str
    command: str = ""  # the command word the message matched, e.g. "!prompt"
    received_at: float = field(default_factory=time.monotonic)


def match_command(message: str, commands: Sequence[str]) -> tuple[str, str] | None:
    """Match a chat message against command words: `(command, text)` or None.

    Accepts `<command> <text>` case-insensitively; a bare command with no
    text is ignored.
    """
    stripped = message.strip()
    lowered = stripped.lower()
    for command in commands:
        if not lowered.startswith(command.lower()):
            continue
        remainder = stripped[len(command):]
        if remainder and not remainder[0].isspace():
            continue  # e.g. "!promptfoo" is not "!prompt foo"
        text = remainder.strip()
        if text:
            return command, text
    return None


class ChatSource(ABC):
    """One chat platform delivering viewer prompts."""

    name: str = "chat"

    @abstractmethod
    async def run(self, on_prompt: Callable[[ChatPrompt], None]) -> None:
        """Watch the chat forever, delivering accepted prompts to the callback."""

    async def close(self) -> None:
        """Release resources. Called once at shutdown; default is a no-op."""
