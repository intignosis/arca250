"""Prompt upsampling: turn a viewer's rough idea into fasth3-ready scenes.

One LLM call per prompt, against any OpenAI-compatible endpoint. The model
picks the shape the idea calls for — one scene of any legal length, or a
chunked short story of up to `max_chunks` short clips with a setup,
development, and payoff — writes each scene as a self-contained
text-to-video prompt in the configured style/character, and picks each
scene's length in seconds.

Why the prompt is written the way it is — these rules come from how fasth3
actually behaves, so keep them intact when editing:

  * **Every scene is an independent clip with no memory.** The single biggest
    quality lever. A scene that says "the same forest" renders a *different*
    forest; each scene must re-describe the entire setting, subjects, light,
    and style from scratch. (Learned on the earlier story-livestream client,
    where under-described scenes visibly "lost" characters between cuts.)
  * **800 characters is the model's hard cap per prompt** (`MAX_PROMPT_CHARS`
    in `fasth3_types.py`); the LLM is told 750 to leave headroom, and
    `_sanitize` hard-truncates anyway, because LLMs do not count characters
    reliably.
  * **Scene length is a real decision, not a constant.** fasth3 accepts
    5.167–14.375 s per clip (the live bounds arrive from `state_update` and
    are injected into the prompt); a single action reads well short, an
    establishing or evolving shot deserves length.
  * **fasth3 renders synchronized audio**, so the prompt asks for a brief
    soundscape line — clips come out flat without it.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# fasth3's enqueue cap (fasth3_types.MAX_PROMPT_CHARS). Hard limit, enforced
# server-side; _sanitize truncates to it.
MAX_PROMPT_CHARS = 800
# What the LLM is asked to stay under, leaving headroom for its poor counting.
_TARGET_PROMPT_CHARS = 750

_SYSTEM_PROMPT = """\
You are the scene director of a live, chat-driven AI video stream. Viewers
send short, rough ideas; you turn each one into one or more polished
text-to-video prompts for a model that generates short clips with
synchronized audio.

STYLE / CHARACTER — every scene is rendered in this identity; weave it into
every scene prompt, never contradict it:
{style}

HOW THE VIDEO MODEL WORKS (hard constraints):
- Each scene becomes ONE independent clip. The model has NO memory between
  clips: every scene prompt must be fully self-contained and re-describe the
  entire setting, subjects, lighting, palette, mood, and style — even when
  nothing changed from the previous scene. Anything you omit will vanish or
  mutate between scenes.
- Each scene prompt must be under {target_chars} characters. This is a hard
  limit; prefer cutting adjectives over cutting subjects or setting.
- Each scene has a duration in seconds, between {min_seconds} and
  {max_seconds}. Choose deliberately: one simple beat reads well around
  {min_seconds}-8 s; a slow reveal, a journey, or a scene with several beats
  deserves 10-{max_seconds} s.
- The model renders picture AND sound. End each scene prompt with one short
  clause of soundscape (ambience, music mood, or effects).
- Describe only what the camera sees and the microphone hears: no text
  overlays, no UI, no scene numbers, no camera jargon the model cannot show.

{scene_count_rules}

WRITING THE SCENE PROMPTS:
- Be concrete and visual: subject, action, setting, camera angle and motion,
  lighting, color palette, atmosphere, then the soundscape clause.
- Strong nouns and verbs over piles of adjectives; vivid but precise.
- Keep the viewer's idea recognizable — enhance it, do not replace it.
- If the idea is hostile, unsafe, or empty, reinterpret it into something
  safe and visually striking in the same spirit instead of refusing.

Reply with ONLY this JSON, nothing else:
{{"title": "short display title for the sequence",
  "scenes": [{{"prompt": "self-contained scene description...", "seconds": 8.0}}]}}
"""

_MULTI_SCENE_RULES = """\
HOW MANY SCENES — two shapes; pick whichever the idea calls for:
- ONE SCENE: a single scene with its length chosen freely in range. Right
  for a mood, a place, a single action or gag. When in doubt, choose this.
- CHUNKED SHORT STORY: 3 to {max_chunks} short chunks — each near the short
  end, roughly {min_seconds}-8 s — that read as one story with a setup, a
  development, and a payoff. Choose this when the idea implies narrative:
  a journey, a transformation, a chase, a day-in-the-life, a punchline
  that needs building up. Two scenes work for a simple before-and-after.
- Never more than {max_chunks} scenes. Do not pad a thin idea into many
  chunks; a story earns its chunks or it is one scene.
- Consecutive scenes play back-to-back as one sequence. Make them feel
  continuous: repeat the shared setting and subjects verbatim enough that
  they read as the same place, and change only what the story moves."""

_SINGLE_SCENE_RULES = """\
HOW MANY SCENES:
- Exactly one scene, with its length chosen freely in range. Distill the
  idea into a single, complete shot."""


@dataclass(frozen=True)
class Scene:
    """One upsampled scene: a prompt fasth3 can take verbatim, and a length."""

    prompt: str
    seconds: float


@dataclass(frozen=True)
class SceneGroup:
    """The scenes one prompt expanded into, played back-to-back.

    ``generated`` marks filler groups made from the idle prompt list rather
    than a viewer request; the director may evict their clips from the
    model's queue to make room for viewer groups.
    """

    group_id: str
    title: str
    author: str
    source: str
    raw_prompt: str
    scenes: list[Scene]
    generated: bool = False


class PromptUpsampler:
    """Expand chat ideas into styled, self-contained fasth3 scenes."""

    def __init__(
        self,
        api_key: str,
        model: str,
        style: str,
        max_chunks: int,
        base_url: str | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._style = style.strip() or "Cinematic, photoreal, rich natural light."
        self._max_chunks = max_chunks

    async def upsample(
        self,
        raw_prompt: str,
        author: str,
        source: str,
        min_seconds: float,
        max_seconds: float,
        generated: bool = False,
        max_chunks: int | None = None,
    ) -> SceneGroup:
        """One idea in, one validated scene group out. Never raises.

        `min_seconds`/`max_seconds` are the live bounds from the model's
        `state_update`, so the LLM always chooses within what the deployment
        actually accepts. `max_chunks` caps this call below the configured
        ceiling (the idle filler passes 1 so its groups stay one-clip and
        evictable). On any LLM failure the raw prompt (styled, truncated)
        becomes a single scene — the stream keeps moving.
        """
        chunk_cap = min(max_chunks or self._max_chunks, self._max_chunks)
        scene_count_rules = (
            _MULTI_SCENE_RULES.format(
                max_chunks=chunk_cap, min_seconds=f"{min_seconds:g}"
            )
            if chunk_cap > 1
            else _SINGLE_SCENE_RULES
        )
        system = _SYSTEM_PROMPT.format(
            style=self._style,
            target_chars=_TARGET_PROMPT_CHARS,
            min_seconds=f"{min_seconds:g}",
            max_seconds=f"{max_seconds:g}",
            scene_count_rules=scene_count_rules,
        )
        group_id = uuid.uuid4().hex[:12]
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Viewer idea: {raw_prompt}"},
                ],
                temperature=0.8,
                max_tokens=1800,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content or "{}")
            title = str(data.get("title") or raw_prompt[:60]).strip()
            scenes = self._validate_scenes(
                data.get("scenes", []), chunk_cap, min_seconds, max_seconds
            )
            if not scenes:
                raise ValueError("no usable scenes in the reply")
        except Exception as error:
            logger.warning("[upsampler] falling back to the raw prompt: %s", error)
            title = raw_prompt[:60]
            scenes = [
                Scene(
                    prompt=_sanitize(f"{self._style}. {raw_prompt}"),
                    seconds=_clamp(8.0, min_seconds, max_seconds),
                )
            ]

        group = SceneGroup(
            group_id=group_id,
            title=title,
            author=author,
            source=source,
            raw_prompt=raw_prompt,
            scenes=scenes,
            generated=generated,
        )
        for index, scene in enumerate(group.scenes, start=1):
            logger.info(
                "[upsampler] %s scene %d/%d (%.1fs): %.100s...",
                group_id, index, len(group.scenes), scene.seconds, scene.prompt,
            )
        return group

    def _validate_scenes(
        self, raw_scenes: list, chunk_cap: int, min_seconds: float, max_seconds: float
    ) -> list[Scene]:
        """Enforce every constraint the LLM was asked for; trust nothing."""
        scenes: list[Scene] = []
        for raw in raw_scenes[:chunk_cap]:
            if not isinstance(raw, dict):
                continue
            prompt = _sanitize(str(raw.get("prompt", "")))
            if not prompt:
                continue
            try:
                seconds = float(raw.get("seconds", 8.0))
            except (TypeError, ValueError):
                seconds = 8.0
            scenes.append(Scene(prompt=prompt, seconds=_clamp(seconds, min_seconds, max_seconds)))
        return scenes


def _sanitize(prompt: str) -> str:
    """Collapse whitespace and hard-truncate to fasth3's prompt cap."""
    collapsed = " ".join(prompt.split())
    return collapsed[:MAX_PROMPT_CHARS].strip()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
