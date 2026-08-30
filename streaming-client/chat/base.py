"""The chat-source contract: where `!prompt` requests come from.

A `ChatSource` watches one chat (a Twitch channel, a YouTube live chat, ...)
and calls the provided callback with a `ChatPrompt` for every message that
starts with the configured command word. Everything else about the platform —
transport, polling cadence, reconnects — is the source's own business.

Rules for implementers:
  * `run` is a long-lived coroutine: connect, deliver prompts, and recover
    from transient failures internally (with backoff). Return only when
    cancelled or the source is permanently unusable (log why).
  * Deliver each message at most once, and nothing from before the source
    started — replaying a chat backlog at startup floods the queue.
  * Strip the command word before delivering; `text` is the bare idea.
  * The callback is synchronous, cheap, and never raises; call it inline.

To add a platform (Kick, Discord, ...): implement this class in a new module,
wire it in `main.py`'s `build_chat_sources`, and document the env vars in
`.env.example` and the README.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChatPrompt:
    """One accepted `!prompt` request from a viewer."""

    source: str  # e.g. "twitch", "youtube"
    author: str
    text: str
    received_at: float = field(default_factory=time.monotonic)


def parse_command(message: str, command: str) -> str | None:
    """Extract the prompt text from a chat message, or None.

    Accepts `<command> <text>` case-insensitively; a bare command with no text
    is ignored.
    """
    stripped = message.strip()
    if not stripped.lower().startswith(command.lower()):
        return None
    remainder = stripped[len(command):]
    if remainder and not remainder[0].isspace():
        return None  # e.g. "!promptfoo" is not "!prompt foo"
    text = remainder.strip()
    return text or None


class ChatSource(ABC):
    """One chat platform delivering viewer prompts."""

    name: str = "chat"

    @abstractmethod
    async def run(self, on_prompt: Callable[[ChatPrompt], None]) -> None:
        """Watch the chat forever, delivering accepted prompts to the callback."""

    async def close(self) -> None:
        """Release resources. Called once at shutdown; default is a no-op."""
