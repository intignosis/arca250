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
  * **The scene-prompt format is distilled from the checkpoint's official
    prompts** (the paper's `integrated_multimodal_description` examples):
    a medium/style declaration first, `[Shot N]` segments with cut
    timestamps ("At 00:04.500, the camera cuts to..."), characters tagged
    `S1`/`S2` with a compact visual identity, the camera as its own
    sentence with amplitude and speed, dialogue inside the
    `<d>[Language] ...</d>` speech marker with the voice described,
    explicit constraint assertions ("no readable signs, captions, or
    logos"), and a closing diegetic soundscape. The model was trained on
    that shape; prompts in it render dramatically better.
  * **Length policy: a single-clip generation always runs the maximum
    length** (the live `clip_seconds_max` from `state_update`), enforced in
    code after validation — its 2-4 internal shots carry the variety. The
    short end of the range is reserved for transition chunks inside
    multi-scene stories (an establishing cut, a reaction beat); content
    chunks run long.
  * **Safety is moderation's job, not this prompt's.** The viewer's idea has
    already passed the moderator by the time it arrives here, so the prompt
    asks for faithful staging and never for softening or reinterpreting the
    idea (see `moderator.py`).
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
# Sized so an overshoot still fits under the 800 hard cap: the sanitizer
# truncates mid-word at 800, and what it cuts is the prompt's tail — the
# soundscape sentence the format deliberately puts last.
_TARGET_PROMPT_CHARS = 700

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
- Each scene prompt must be under {target_chars} characters. Anything past
  the limit is CUT OFF mid-sentence — and the cut lands on your closing
  soundscape. Prefer cutting adjectives over subjects or setting.
- Describe only what the camera sees and the microphone hears: no UI, no
  scene numbers, nothing the model cannot show.

{scene_count_rules}

SCENE PROMPT FORMAT — the model was trained on prompts with this exact
shape; follow it inside every scene prompt, compressed to the char budget:
- Open with the medium and style, then the setting: "[Shot 1] <style>, 16:9
  widescreen. <place, light, mood>." The style block above IS that medium.
- Give each character a compact visual identity at first mention and a tag —
  "the doughy dad in a too-tight polo (S1)" — then refer back by tag. At
  most two speaking characters per scene.
- Action first, in concrete physical beats; then the camera as its own
  sentence with movement, angle, amplitude and speed: "The camera arcs
  around the pair with medium amplitude at fast speed."
- Split the scene into 2-4 shots. Every later shot opens "[Shot N] At
  00:0X.000, the camera cuts to <view> as <action>", with timestamps spread
  across the scene's duration.
- Dialogue uses the model's speech marker, never plain narration:
  S1 shouts in a hoarse, panicked dad voice: <d>[English] It's alive!</d>
  Name the speaker's tag, describe the voice, and put the exact words
  inside <d>[Language] ...</d>. Do not paraphrase speech the viewer asked
  for, and never put words outside the marker.
- State the constraints that must hold as explicit facts in the prompt:
  "no readable signs, captions, or logos", "their hands stay empty
  throughout" — the model honours what the prompt asserts.
- End with one or two sentences of soundscape: the diegetic sounds
  synchronized to the actions (foley, impacts, breathing, room tone), and
  music only when wanted, named by instrument and mood.

WRITING THE SCENE PROMPTS:
- Strong nouns and verbs over piles of adjectives; vivid but precise.
- Keep the viewer's idea recognizable — enhance it, do not replace it. The
  idea has already passed moderation before it reaches you; your job is
  faithful staging, not policing.

Reply with ONLY this JSON, nothing else:
{{"title": "short display title for the sequence",
  "scenes": [{{"prompt": "self-contained scene description...", "seconds": 8.0}}]}}
The "scenes" array is REQUIRED even when it holds a single scene; never
flatten a scene's fields to the top level.
"""

_MULTI_SCENE_RULES = """\
HOW MANY SCENES, AND HOW LONG — two shapes; pick what the idea calls for:
- ONE SCENE: a single clip that ALWAYS runs the full {max_seconds} seconds —
  never shorter — with its 2-4 internal shots spread across that time.
  Right for a mood, a place, a single action or gag. When in doubt, this.
- CHUNKED SHORT STORY: 3 to {max_chunks} chunks that read as one story with
  a setup, a development, and a payoff. Content chunks run 8-{max_seconds}
  seconds; the short end ({min_seconds}-8 s) is ONLY for transitions — an
  establishing cut, a reaction beat, a snap punchline — never for a chunk
  that carries the story. Choose this shape when the idea implies
  narrative: a journey, a transformation, a chase, a build-up.
- Never more than {max_chunks} scenes. Do not pad a thin idea into many
  chunks; a story earns its chunks or it is one full-length scene.
- Consecutive scenes play back-to-back as one sequence. Make them feel
  continuous: repeat the shared setting and subjects verbatim enough that
  they read as the same place, and change only what the story moves."""

_SINGLE_SCENE_RULES = """\
HOW MANY SCENES, AND HOW LONG:
- Exactly one scene, and it ALWAYS runs the full {max_seconds} seconds.
  Distill the idea into one complete arc of 2-4 internal shots spread
  across that time."""


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
                max_chunks=chunk_cap,
                min_seconds=f"{min_seconds:g}",
                max_seconds=f"{max_seconds:g}",
            )
            if chunk_cap > 1
            else _SINGLE_SCENE_RULES.format(max_seconds=f"{max_seconds:g}")
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
            content = response.choices[0].message.content or ""
            data = json.loads(content or "{}")
            title = str(data.get("title") or raw_prompt[:60]).strip()
            raw_scenes = data.get("scenes")
            if isinstance(raw_scenes, dict):
                raw_scenes = [raw_scenes]
            if not raw_scenes and "prompt" in data:
                # Some models flatten a single scene's fields to the top
                # level despite the schema; accept it as one scene.
                raw_scenes = [data]
            scenes = self._validate_scenes(
                raw_scenes or [], chunk_cap, min_seconds, max_seconds
            )
            if not scenes:
                raise ValueError(
                    "no usable scenes in the reply "
                    f"(finish={response.choices[0].finish_reason}, "
                    f"head={content[:200]!r})"
                )
            if len(scenes) == 1:
                # A single-clip generation always runs the maximum length;
                # short clips are reserved for transition chunks in stories.
                scenes = [Scene(prompt=scenes[0].prompt, seconds=max_seconds)]
        except Exception as error:
            logger.warning("[upsampler] falling back to the raw prompt: %s", error)
            title = raw_prompt[:60]
            # The viewer's idea gets the char budget first; the style fills
            # whatever remains (a long STYLE must never truncate the idea away).
            idea = _sanitize(raw_prompt)
            style_room = MAX_PROMPT_CHARS - len(idea) - 2
            fallback = f"{idea}. {self._style[:style_room]}" if style_room > 20 else idea
            scenes = [
                # A single clip, so it takes the maximum length like every
                # other one-scene generation.
                Scene(prompt=_sanitize(fallback), seconds=max_seconds)
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
    """Collapse whitespace and fit under fasth3's prompt cap, ending clean.

    LLMs overshoot the character target they are given, and a blind cut at
    the cap ends the prompt mid-word — worse for the model than losing the
    final sentence. Over-long prompts are therefore cut at the last sentence
    boundary (or speech-marker close) that fits; the mid-word cut remains
    only as the last resort for a prompt written as one giant sentence.
    """
    collapsed = " ".join(prompt.split())
    if len(collapsed) <= MAX_PROMPT_CHARS:
        return collapsed.strip()
    head = collapsed[:MAX_PROMPT_CHARS]
    boundary = max(
        head.rfind(". "), head.rfind("! "), head.rfind("? "),
        head.rfind(".</d>") + 4 if head.rfind(".</d>") != -1 else -1,
        head.rfind("</d>") + 3 if head.rfind("</d>") != -1 else -1,
    )
    if boundary > MAX_PROMPT_CHARS // 2:
        return head[: boundary + 1].strip()
    return head.strip()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
