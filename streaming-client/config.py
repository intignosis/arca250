"""Configuration for the fast-h3 streaming client.

Everything comes from the environment (a `.env` file is loaded when present),
with a handful of CLI overrides for the things you flip per run. `Config.load`
is the only reader; the rest of the client takes a `Config` and never touches
`os.environ`.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _flag(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _load_preset(name_or_path: str) -> dict:
    """Load and validate one preset: the creative bundle the stream runs.

    `PRESET` names a file in the `presets/` folder next to this module
    (`default` → `presets/default.json`); a value containing a path
    separator or ending in `.json` is used as an explicit path instead.

    The format, all of it:
        {
          "name":        "display name (optional)",
          "description": "what this preset is (optional)",
          "style":       "the style/character block every upsampled scene
                          is written in (required, non-empty)",
          "idle_prompts": ["premade prompt", ...]  (required; may be empty,
                          which disables the idle filler)
        }
    """
    if "/" in name_or_path or name_or_path.endswith(".json"):
        path = Path(name_or_path)
    else:
        path = Path(__file__).parent / "presets" / f"{name_or_path}.json"
    if not path.is_file():
        raise SystemExit(f"preset not found: {path} (set PRESET or --preset)")
    try:
        preset = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"preset {path} is not valid JSON: {error}") from None
    style = preset.get("style")
    prompts = preset.get("idle_prompts")
    if not isinstance(style, str) or not style.strip():
        raise SystemExit(f"preset {path} needs a non-empty string `style`")
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        raise SystemExit(f"preset {path} needs `idle_prompts` as a list of strings")
    return {
        "style": style.strip(),
        "idle_prompts": [p.strip() for p in prompts if p.strip()],
    }


@dataclass(frozen=True)
class Config:
    """One immutable snapshot of everything the client is configured with."""

    # Reactor
    model: str
    api_key: str | None
    local: bool
    local_url: str

    # Upsampling
    openai_api_key: str
    openai_base_url: str | None
    openai_model: str
    max_chunks: int

    # Preset: the creative bundle (style + premade idle prompts)
    preset_name: str
    style: str
    idle_prompts: tuple[str, ...]

    # Moderation (its own endpoint: the upsampling gateway may not expose
    # /moderations, so this can point at api.openai.com while upsampling
    # goes elsewhere)
    moderation_enabled: bool
    moderation_api_key: str
    moderation_base_url: str | None
    moderation_model: str

    # Idle filler
    idle_queue_target: int

    # Overlay (which overlay runs is code, main.py; this only switches it)
    overlay_enabled: bool

    # Sink
    sink: str  # "rtmp" | "noop"
    rtmp_url: str | None
    rtmp_video_bitrate_k: int

    # Chat
    chat_command: str
    chat_cooldown_s: float
    twitch_channel: str | None
    youtube_video_id: str | None
    youtube_api_key: str | None

    @staticmethod
    def load(argv: list[str] | None = None) -> "Config":
        """Read `.env` + environment, apply CLI overrides, and validate."""
        parser = argparse.ArgumentParser(
            description="Chat-driven fast-h3 livestream client (see README.md)."
        )
        parser.add_argument("--env-file", default=None, help="path to a .env file")
        parser.add_argument("--model", default=None, help="override REACTOR_MODEL")
        parser.add_argument("--api-key", default=None, help="override REACTOR_API_KEY")
        parser.add_argument(
            "--local", action="store_true", help="drive a local `reactor run` (no key)"
        )
        parser.add_argument(
            "--local-url", default=None,
            help="local runtime URL (default REACTOR_LOCAL_URL or http://localhost:8080)",
        )
        parser.add_argument("--sink", default=None, choices=("rtmp", "noop"))
        parser.add_argument("--rtmp-url", default=None, help="override RTMP_URL")
        parser.add_argument("--preset", default=None, help="override PRESET")
        args = parser.parse_args(argv)

        if args.env_file:
            load_dotenv(args.env_file, override=True)
        else:
            load_dotenv()  # ./.env when present; no-op otherwise

        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        openai_base_url = os.environ.get("OPENAI_BASE_URL") or None

        preset_name = args.preset or os.environ.get("PRESET", "default")
        preset = _load_preset(preset_name)

        config = Config(
            model=args.model or os.environ.get("REACTOR_MODEL", "fast-h3"),
            api_key=args.api_key or os.environ.get("REACTOR_API_KEY") or None,
            local=args.local or _flag(os.environ.get("REACTOR_LOCAL")),
            local_url=args.local_url
            or os.environ.get("REACTOR_LOCAL_URL")
            or "http://localhost:8080",
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            max_chunks=max(1, int(os.environ.get("MAX_CHUNKS", "6"))),
            preset_name=preset_name,
            style=preset["style"],
            idle_prompts=tuple(preset["idle_prompts"]),
            moderation_enabled=_flag(os.environ.get("MODERATION_ENABLED", "1")),
            moderation_api_key=os.environ.get("MODERATION_API_KEY") or openai_api_key,
            moderation_base_url=os.environ.get("MODERATION_BASE_URL") or openai_base_url,
            moderation_model=os.environ.get("MODERATION_MODEL", "omni-moderation-latest"),
            idle_queue_target=int(os.environ.get("IDLE_QUEUE_TARGET", "6")),
            overlay_enabled=_flag(os.environ.get("OVERLAY_ENABLED", "1")),
            sink=(args.sink or os.environ.get("SINK", "noop")).lower(),
            rtmp_url=args.rtmp_url or os.environ.get("RTMP_URL") or None,
            rtmp_video_bitrate_k=int(os.environ.get("RTMP_VIDEO_BITRATE_K", "4500")),
            chat_command=os.environ.get("CHAT_COMMAND", "!prompt").strip(),
            chat_cooldown_s=float(os.environ.get("CHAT_COOLDOWN_S", "30")),
            twitch_channel=os.environ.get("TWITCH_CHANNEL") or None,
            youtube_video_id=os.environ.get("YOUTUBE_VIDEO_ID") or None,
            youtube_api_key=os.environ.get("YOUTUBE_API_KEY") or None,
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Fail fast on contradictions instead of half-starting."""
        if not self.local and not self.api_key:
            raise SystemExit(
                "Either REACTOR_API_KEY (hosted) or --local / REACTOR_LOCAL=1 is required."
            )
        if not self.openai_api_key:
            raise SystemExit("OPENAI_API_KEY is required for prompt upsampling.")
        if self.sink == "rtmp" and not self.rtmp_url:
            raise SystemExit("SINK=rtmp needs RTMP_URL (including the stream key).")
        if self.sink not in ("rtmp", "noop"):
            raise SystemExit(f"Unknown SINK {self.sink!r}; use rtmp or noop.")
        if self.youtube_video_id and not self.youtube_api_key:
            raise SystemExit("YOUTUBE_VIDEO_ID needs YOUTUBE_API_KEY.")
        if not self.chat_command.startswith("!"):
            raise SystemExit("CHAT_COMMAND should start with '!' (e.g. !prompt).")
