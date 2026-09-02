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


class PresetError(ValueError):
    """A preset file is missing or malformed."""


def presets_dir() -> Path:
    """The `presets/` folder next to this module."""
    return Path(__file__).parent / "presets"


def available_presets() -> list[str]:
    """Preset names loadable right now, read fresh from the folder.

    Scanned on every call, so a JSON dropped into `presets/` mid-run is
    immediately switchable (see `admin.py`'s `!switch`).
    """
    return sorted(path.stem for path in presets_dir().glob("*.json"))


def load_preset(name_or_path: str) -> dict:
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

    Unknown keys are ignored, so presets can carry their own notes.

    Raises:
        PresetError: when the file is missing, not JSON, or missing a
            required key. Startup turns this into a clean exit; the admin
            `!switch` path logs it and keeps the current preset.
    """
    if "/" in name_or_path or name_or_path.endswith(".json"):
        path = Path(name_or_path)
    else:
        path = presets_dir() / f"{name_or_path}.json"
    if not path.is_file():
        raise PresetError(f"preset not found: {path}")
    try:
        preset = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PresetError(f"preset {path} is not valid JSON: {error}") from None
    style = preset.get("style")
    prompts = preset.get("idle_prompts")
    if not isinstance(style, str) or not style.strip():
        raise PresetError(f"preset {path} needs a non-empty string `style`")
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        raise PresetError(f"preset {path} needs `idle_prompts` as a list of strings")
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

    # Upsampling (Claude, via the Anthropic SDK)
    anthropic_api_key: str
    anthropic_base_url: str | None
    anthropic_model: str
    max_chunks: int

    # Preset: the creative bundle (style + premade idle prompts)
    preset_name: str
    style: str
    idle_prompts: tuple[str, ...]

    # Moderation. A separate provider on purpose: Anthropic has no
    # moderations endpoint, so upsampling runs on Claude while the
    # fail-closed safety classifier stays on OpenAI's /moderations.
    moderation_enabled: bool
    moderation_api_key: str
    moderation_base_url: str | None
    moderation_model: str

    # Idle filler
    idle_queue_target: int

    # Overlay (which overlay runs is code, main.py; this only switches it)
    overlay_enabled: bool

    # Rehearsal: broadcast without a model (delivery-path test).
    rehearse: bool

    # Sink
    sink: str  # "rtmp" | "noop"
    rtmp_url: str | None
    rtmp_video_bitrate_k: int
    # Music bed, mixed under the scene audio by the rtmp sink (ducked so
    # dialogue and drops stay in front). Empty disables it.
    music_path: str | None
    music_volume: float

    # Chat
    chat_command: str
    chat_cooldown_s: float
    twitch_channel: str | None
    youtube_video_id: str | None
    youtube_api_key: str | None

    # Admins: chat usernames allowed to send admin commands (see admin.py).
    # Normalized lowercase; entries are bare names or `source:name`.
    admin_users: frozenset[str]

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
        parser.add_argument(
            "--rehearse", action="store_true",
            help="start broadcasting immediately with the default canvas and "
                 "no model: black frames, overlay, and the music bed — for "
                 "proving the delivery path end to end",
        )
        parser.add_argument("--rtmp-url", default=None, help="override RTMP_URL")
        parser.add_argument("--preset", default=None, help="override PRESET")
        args = parser.parse_args(argv)

        if args.env_file:
            load_dotenv(args.env_file, override=True)
        else:
            load_dotenv()  # ./.env when present; no-op otherwise

        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        anthropic_base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        openai_base_url = os.environ.get("OPENAI_BASE_URL") or None

        preset_name = args.preset or os.environ.get("PRESET", "default")
        try:
            preset = load_preset(preset_name)
        except PresetError as error:
            raise SystemExit(f"{error} (set PRESET or --preset)") from None

        config = Config(
            model=args.model or os.environ.get("REACTOR_MODEL", "fast-h3"),
            api_key=args.api_key or os.environ.get("REACTOR_API_KEY") or None,
            local=args.local or _flag(os.environ.get("REACTOR_LOCAL")),
            local_url=args.local_url
            or os.environ.get("REACTOR_LOCAL_URL")
            or "http://localhost:8080",
            anthropic_api_key=anthropic_api_key,
            anthropic_base_url=anthropic_base_url,
            anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-5"),
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
            rehearse=bool(getattr(args, "rehearse", False)),
            rtmp_url=args.rtmp_url or os.environ.get("RTMP_URL") or None,
            rtmp_video_bitrate_k=int(os.environ.get("RTMP_VIDEO_BITRATE_K", "4500")),
            music_path=os.environ.get("MUSIC_PATH") or None,
            music_volume=float(os.environ.get("MUSIC_VOLUME", "0.35")),
            chat_command=os.environ.get("CHAT_COMMAND", "!prompt").strip(),
            chat_cooldown_s=float(os.environ.get("CHAT_COOLDOWN_S", "30")),
            twitch_channel=os.environ.get("TWITCH_CHANNEL") or None,
            youtube_video_id=os.environ.get("YOUTUBE_VIDEO_ID") or None,
            youtube_api_key=os.environ.get("YOUTUBE_API_KEY") or None,
            admin_users=frozenset(
                entry.strip().lower()
                for entry in os.environ.get("ADMIN_USERS", "").split(",")
                if entry.strip()
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Fail fast on contradictions instead of half-starting."""
        if not self.local and not self.api_key:
            raise SystemExit(
                "Either REACTOR_API_KEY (hosted) or --local / REACTOR_LOCAL=1 is required."
            )
        if not self.anthropic_api_key:
            raise SystemExit("ANTHROPIC_API_KEY is required for prompt upsampling.")
        if self.moderation_enabled and not self.moderation_api_key:
            raise SystemExit(
                "MODERATION_API_KEY (an OpenAI key) is required while "
                "MODERATION_ENABLED=1 — Anthropic has no moderations endpoint. "
                "Set MODERATION_ENABLED=0 only if you accept an unmoderated stream."
            )
        if self.sink == "rtmp" and not self.rtmp_url:
            raise SystemExit("SINK=rtmp needs RTMP_URL (including the stream key).")
        if self.sink not in ("rtmp", "noop"):
            raise SystemExit(f"Unknown SINK {self.sink!r}; use rtmp or noop.")
        if self.youtube_video_id and not self.youtube_api_key:
            raise SystemExit("YOUTUBE_VIDEO_ID needs YOUTUBE_API_KEY.")
        if not self.chat_command.startswith("!"):
            raise SystemExit("CHAT_COMMAND should start with '!' (e.g. !prompt).")
