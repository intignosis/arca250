"""Configuration for the fasth3 streaming client.

Everything comes from the environment (a `.env` file is loaded when present),
with a handful of CLI overrides for the things you flip per run. `Config.load`
is the only reader; the rest of the client takes a `Config` and never touches
`os.environ`.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _flag(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _load_prompt_lines(path: Path, required: bool) -> tuple[str, ...]:
    """Read one prompt per line, skipping blanks and `#` comments."""
    if not path.is_file():
        if required:
            raise SystemExit(f"IDLE_PROMPTS_FILE not found: {path}")
        return ()
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


@dataclass(frozen=True)
class Config:
    """One immutable snapshot of everything the client is configured with."""

    # Reactor
    model: str
    api_key: str | None
    local: bool

    # Upsampling
    openai_api_key: str
    openai_base_url: str | None
    openai_model: str
    style: str
    max_chunks: int

    # Moderation (its own endpoint: the upsampling gateway may not expose
    # /moderations, so this can point at api.openai.com while upsampling
    # goes elsewhere)
    moderation_enabled: bool
    moderation_api_key: str
    moderation_base_url: str | None
    moderation_model: str

    # Idle filler
    idle_prompts: tuple[str, ...]
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
            description="Chat-driven fasth3 livestream client (see README.md)."
        )
        parser.add_argument("--env-file", default=None, help="path to a .env file")
        parser.add_argument("--model", default=None, help="override REACTOR_MODEL")
        parser.add_argument("--api-key", default=None, help="override REACTOR_API_KEY")
        parser.add_argument(
            "--local", action="store_true", help="drive a local `reactor run` (no key)"
        )
        parser.add_argument("--sink", default=None, choices=("rtmp", "noop"))
        parser.add_argument("--rtmp-url", default=None, help="override RTMP_URL")
        args = parser.parse_args(argv)

        if args.env_file:
            load_dotenv(args.env_file, override=True)
        else:
            load_dotenv()  # ./.env when present; no-op otherwise

        style = os.environ.get("STYLE", "").strip()
        style_file = os.environ.get("STYLE_FILE", "").strip()
        if style_file:
            style = Path(style_file).read_text(encoding="utf-8").strip()

        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        openai_base_url = os.environ.get("OPENAI_BASE_URL") or None

        idle_file = os.environ.get("IDLE_PROMPTS_FILE", "").strip()
        if idle_file:
            idle_prompts = _load_prompt_lines(Path(idle_file), required=True)
        else:
            # The list shipped next to this file; absent means filler off.
            default_file = Path(__file__).parent / "idle_prompts.txt"
            idle_prompts = _load_prompt_lines(default_file, required=False)

        config = Config(
            model=args.model or os.environ.get("REACTOR_MODEL", "fasth3"),
            api_key=args.api_key or os.environ.get("REACTOR_API_KEY") or None,
            local=args.local or _flag(os.environ.get("REACTOR_LOCAL")),
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            style=style,
            max_chunks=max(1, int(os.environ.get("MAX_CHUNKS", "6"))),
            moderation_enabled=_flag(os.environ.get("MODERATION_ENABLED", "1")),
            moderation_api_key=os.environ.get("MODERATION_API_KEY") or openai_api_key,
            moderation_base_url=os.environ.get("MODERATION_BASE_URL") or openai_base_url,
            moderation_model=os.environ.get("MODERATION_MODEL", "omni-moderation-latest"),
            idle_prompts=idle_prompts,
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
