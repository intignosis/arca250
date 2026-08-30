"""Twitch chat over anonymous IRC.

Twitch lets read-only clients join chat without credentials: connect to
`irc.chat.twitch.tv:6697` (TLS), send `NICK justinfan<digits>`, `JOIN
#channel`, and PRIVMSG lines flow. No OAuth app, no token rotation — which is
exactly right for a client that only reads `!prompt` messages.

The one protocol obligation is answering `PING` with `PONG`, or the server
drops the connection after ~5 minutes. Everything else here is line parsing
and a reconnect loop with backoff.
"""

from __future__ import annotations

import asyncio
import logging
import random
import ssl
from collections.abc import Callable

from .base import ChatPrompt, ChatSource, parse_command

logger = logging.getLogger(__name__)

_HOST = "irc.chat.twitch.tv"
_PORT = 6697
_RECONNECT_MIN_S = 2.0
_RECONNECT_MAX_S = 60.0


class TwitchChat(ChatSource):
    """Read `!prompt` messages from one Twitch channel, anonymously."""

    name = "twitch"

    def __init__(self, channel: str, command: str) -> None:
        self._channel = channel.lstrip("#").lower()
        self._command = command

    async def run(self, on_prompt: Callable[[ChatPrompt], None]) -> None:
        backoff = _RECONNECT_MIN_S
        while True:
            try:
                await self._session(on_prompt)
                backoff = _RECONNECT_MIN_S  # a session that ran resets backoff
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning("[twitch] connection lost: %s", error)
            logger.info("[twitch] reconnecting in %.0fs", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_MAX_S)

    async def _session(self, on_prompt: Callable[[ChatPrompt], None]) -> None:
        reader, writer = await asyncio.open_connection(
            _HOST, _PORT, ssl=ssl.create_default_context()
        )
        try:
            nick = f"justinfan{random.randint(10_000, 99_999)}"
            writer.write(f"NICK {nick}\r\nJOIN #{self._channel}\r\n".encode())
            await writer.drain()
            logger.info("[twitch] joined #%s as %s (read-only)", self._channel, nick)

            while True:
                raw = await reader.readline()
                if not raw:
                    raise ConnectionError("server closed the connection")
                line = raw.decode(errors="replace").rstrip("\r\n")

                if line.startswith("PING"):
                    writer.write(line.replace("PING", "PONG", 1).encode() + b"\r\n")
                    await writer.drain()
                    continue

                prompt = self._parse_privmsg(line)
                if prompt is not None:
                    logger.info("[twitch] %s: %s", prompt.author, prompt.text)
                    on_prompt(prompt)
        finally:
            writer.close()

    def _parse_privmsg(self, line: str) -> ChatPrompt | None:
        """Parse `:nick!user@host PRIVMSG #chan :message` into a prompt."""
        if " PRIVMSG " not in line or not line.startswith(":"):
            return None
        prefix, _, rest = line[1:].partition(" PRIVMSG ")
        author = prefix.split("!", 1)[0]
        _, _, message = rest.partition(" :")
        text = parse_command(message, self._command)
        if text is None:
            return None
        return ChatPrompt(source=self.name, author=author, text=text)
