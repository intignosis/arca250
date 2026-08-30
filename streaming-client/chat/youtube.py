"""YouTube live chat via the Data API v3.

There is no push transport for YouTube chat, so this polls
`liveChat/messages` at the interval the API itself asks for
(`pollingIntervalMillis`). Two learnings from the earlier livestream client
are kept:

  * The `liveChatId` is resolved from the *video id* of the running broadcast
    (`videos?part=liveStreamingDetails`) — a video that is not currently live
    has no `activeLiveChatId`, which is reported instead of silently polling
    nothing.
  * The first poll returns recent history. It is consumed silently so a
    restart never replays a backlog of old `!prompt` messages into the queue.

Quota note: each poll costs Data API quota (default 10k units/day). At the
~6 s interval YouTube asks for, a day of streaming fits comfortably, but do
not shorten the interval below what the API returns.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence

import aiohttp

from .base import ChatPrompt, ChatSource, match_command

logger = logging.getLogger(__name__)

_API = "https://www.googleapis.com/youtube/v3"
_RETRY_S = 30.0


class YouTubeChat(ChatSource):
    """Read command messages from one live broadcast's chat."""

    name = "youtube"

    def __init__(self, video_id: str, api_key: str, commands: Sequence[str]) -> None:
        self._video_id = video_id
        self._api_key = api_key
        self._commands = tuple(commands)
        self._session: aiohttp.ClientSession | None = None
        self._seen_ids: set[str] = set()

    async def run(self, on_prompt: Callable[[ChatPrompt], None]) -> None:
        self._session = aiohttp.ClientSession()
        try:
            chat_id = None
            while chat_id is None:
                chat_id = await self._resolve_chat_id()
                if chat_id is None:
                    await asyncio.sleep(_RETRY_S)

            page_token: str | None = None
            first_poll = True
            interval = 6.0
            while True:
                try:
                    items, page_token, interval = await self._poll(chat_id, page_token)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    logger.warning("[youtube] poll failed: %s", error)
                    await asyncio.sleep(_RETRY_S)
                    continue

                for item in items:
                    prompt = self._to_prompt(item)
                    if prompt is None:
                        continue
                    if first_poll:
                        continue  # history from before we started
                    logger.info("[youtube] %s: %s", prompt.author, prompt.text)
                    on_prompt(prompt)
                first_poll = False
                await asyncio.sleep(interval)
        finally:
            await self.close()

    async def _resolve_chat_id(self) -> str | None:
        assert self._session is not None
        url = f"{_API}/videos"
        params = {
            "part": "liveStreamingDetails",
            "id": self._video_id,
            "key": self._api_key,
        }
        async with self._session.get(url, params=params) as response:
            if response.status != 200:
                body = await response.text()
                logger.error(
                    "[youtube] video lookup failed (HTTP %d): %.200s",
                    response.status, body,
                )
                return None
            data = await response.json()
        items = data.get("items", [])
        if not items:
            logger.error("[youtube] video not found: %s", self._video_id)
            return None
        chat_id = items[0].get("liveStreamingDetails", {}).get("activeLiveChatId")
        if not chat_id:
            logger.warning(
                "[youtube] no active live chat on %s — is the broadcast live? retrying",
                self._video_id,
            )
            return None
        logger.info("[youtube] connected to live chat of %s", self._video_id)
        return chat_id

    async def _poll(
        self, chat_id: str, page_token: str | None
    ) -> tuple[list[dict], str | None, float]:
        """One `liveChat/messages` page: (items, next token, requested interval)."""
        assert self._session is not None
        params = {
            "liveChatId": chat_id,
            "part": "snippet,authorDetails",
            "key": self._api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        async with self._session.get(f"{_API}/liveChat/messages", params=params) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {body[:200]}")
            data = await response.json()
        interval = max(data.get("pollingIntervalMillis", 6000) / 1000.0, 2.0)
        return data.get("items", []), data.get("nextPageToken"), interval

    def _to_prompt(self, item: dict) -> ChatPrompt | None:
        message_id = item.get("id", "")
        if not message_id or message_id in self._seen_ids:
            return None
        self._seen_ids.add(message_id)
        if len(self._seen_ids) > 4000:
            # Ids only need to cover the overlap between adjacent polls.
            self._seen_ids = set(list(self._seen_ids)[-2000:])
        snippet = item.get("snippet", {})
        message = snippet.get("displayMessage") or snippet.get(
            "textMessageDetails", {}
        ).get("messageText", "")
        matched = match_command(message, self._commands)
        if matched is None:
            return None
        command, text = matched
        author = item.get("authorDetails", {}).get("displayName", "?")
        return ChatPrompt(source=self.name, author=author, text=text, command=command)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
