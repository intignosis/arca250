"""Admin chat commands: live control of the stream from trusted chatters.

Admins are the chat usernames listed in `ADMIN_USERS` (comma-separated,
matched case-insensitively). A bare entry (`name`) matches that username on
any platform; a scoped entry (`twitch:name`, `youtube:name`) matches it on
one — use scoping when the same display name could be different people on
different platforms.

Admin commands ride the same chat sources as viewer prompts. The router in
`main.py` hands every matched command here first, before the director sees
it, so an admin command never costs a cooldown slot, a moderation call, or
an LLM call — and a `!prompt` from an admin is still just a prompt. A
recognized admin command from a non-admin is consumed and logged, never
treated as a prompt.

The command set:

  * `!switch <preset>` — swap the creative preset live. The name is resolved
    against the `presets/` folder at that moment, so dropping a new JSON
    into the folder makes it switchable with no restart. The upsampler's
    style and the idle filler's prompt list change immediately; clips
    already queued keep the style they were written in and drain naturally.
"""

from __future__ import annotations

import logging

from chat import ChatPrompt
from config import PresetError, available_presets, load_preset
from director import Director
from upsampler import PromptUpsampler

logger = logging.getLogger(__name__)

_SWITCH_COMMAND = "!switch"


class AdminControl:
    """Recognize and execute admin commands arriving from chat."""

    #: Every command word this handler owns; chat sources match on these
    #: (plus the viewer prompt command) so the router can dispatch here.
    commands: tuple[str, ...] = (_SWITCH_COMMAND,)

    def __init__(
        self,
        admin_users: frozenset[str],
        upsampler: PromptUpsampler,
        director: Director,
    ) -> None:
        self._admins = admin_users
        self._upsampler = upsampler
        self._director = director

    def is_admin(self, prompt: ChatPrompt) -> bool:
        """Whether the message's author is on the admin list."""
        author = prompt.author.lower()
        return author in self._admins or f"{prompt.source}:{author}" in self._admins

    def handle(self, prompt: ChatPrompt) -> bool:
        """Execute one admin command; True when the message was one.

        False means the message is not an admin command and should continue
        to the director as a viewer prompt. Synchronous and never raises —
        it is called inline from the chat sources' callback.
        """
        if prompt.command not in self.commands:
            return False
        if not self.is_admin(prompt):
            logger.info(
                "[admin] ignoring %s from non-admin %s@%s",
                prompt.command, prompt.author, prompt.source,
            )
            return True
        if prompt.command == _SWITCH_COMMAND:
            self._switch_preset(prompt)
        return True

    def _switch_preset(self, prompt: ChatPrompt) -> None:
        """`!switch <preset>`: re-point style + idle prompts at another preset."""
        name = prompt.text.split()[0]
        # Only bare names from the presets/ folder — never paths, so chat
        # input cannot point the loader at an arbitrary file.
        if name not in available_presets():
            logger.warning(
                "[admin] %s@%s: unknown preset %r (available: %s)",
                prompt.author, prompt.source, name,
                ", ".join(available_presets()) or "none",
            )
            return
        try:
            preset = load_preset(name)
        except PresetError as error:
            logger.warning(
                "[admin] %s@%s: preset %r failed to load: %s — keeping the "
                "current one",
                prompt.author, prompt.source, name, error,
            )
            return
        self._upsampler.set_style(preset["style"])
        self._director.set_idle_prompts(preset["idle_prompts"])
        logger.info(
            "[admin] %s@%s switched preset to %r (%d idle prompts)",
            prompt.author, prompt.source, name, len(preset["idle_prompts"]),
        )
