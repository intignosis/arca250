"""Moderation for viewer prompts, via the OpenAI moderations API.

Runs on its own endpoint and key (`MODERATION_*` in the environment), falling
back to the upsampling credentials when unset — because the two are often
*not* the same service: an OpenAI-compatible inference gateway typically does
not expose `/moderations` (Reactor's corp gateway answers it with
`provider_not_allowed`), so moderation usually points at api.openai.com while
upsampling goes through the gateway.

Policy decisions, deliberate:
  * Only the viewer's raw prompt is checked, and this is the **only** safety
    gate: the upsampler deliberately stages ideas faithfully rather than
    softening them, so what passes here is what gets rendered. Idle-filler
    prompts are a curated list in this repo and skip the check.
  * Errors **fail closed**: a prompt that cannot be checked is rejected. A
    silent fail-open would quietly turn moderation off exactly when the
    endpoint misbehaves. If the stream must keep accepting prompts without a
    working moderation endpoint, disable moderation explicitly
    (`MODERATION_ENABLED=0`) — that state is then visible in the startup log
    instead of hidden in per-prompt errors.
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class Moderator:
    """Answer "may this viewer prompt drive the stream?" for the director."""

    def __init__(
        self,
        api_key: str,
        model: str,
        enabled: bool,
        base_url: str | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self.enabled = enabled

    async def review(self, text: str) -> str | None:
        """Return None when the text is allowed, else a short rejection reason."""
        if not self.enabled:
            return None
        try:
            response = await self._client.moderations.create(
                model=self._model, input=text
            )
            result = response.results[0]
        except Exception as error:
            logger.error("[moderation] check failed (rejecting prompt): %s", error)
            return "moderation unavailable"
        if not result.flagged:
            return None
        flagged = [
            category
            for category, hit in result.categories.model_dump().items()
            if hit
        ]
        return "flagged: " + ", ".join(flagged) if flagged else "flagged"
