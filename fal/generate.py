"""Generate DASHWORLD test clips on fal, in the preset's own voice.

The staging half of the pipeline is already written and already tuned: the
streaming client's `PromptUpsampler` turns a rough idea into scenes written
in the DASHWORLD style block, and `Moderator` gates viewer input. This
reuses both, then sends each scene to MiniMax H3 with the DASH LoRA and an
optional first frame instead of to fast-h3's clip queue.

What that buys over the fast-h3 path: the character is anchored — a trained
LoRA plus a real reference frame — and it renders with synchronised audio.
What it costs: $0.075 per second of 768p video, billed per generation, so
this is the way to judge the creative work, not the way to run a channel.

    python fal/generate.py --idle 3 --lora fal/dash_lora.json --confirm
    python fal/generate.py --idea "dash plays the rooftop show" --confirm

A bare run prints the scenes and the price and generates nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.request
from pathlib import Path

import fal_client
from dotenv import load_dotenv

CLIENT = Path(__file__).resolve().parent.parent / "streaming-client"
sys.path.insert(0, str(CLIENT))

import config  # noqa: E402
from upsampler import PromptUpsampler  # noqa: E402

MODEL = "minimax/h3/image-to-video/lora"
# $ per second of generated video, by the model's resolution enum.
COST_PER_SECOND = {"480P": 0.0625, "768P": 0.075, "2K": 0.1625, "4K": 0.20}
# fal takes whole seconds; fast-h3's legal range does not apply here.
MIN_SECONDS, MAX_SECONDS = 5, 10


def load_lora(spec: str, scale: float) -> list[dict]:
    """A `loras` entry from either a trainer output file or a bare URL."""
    path = Path(spec)
    url = json.loads(path.read_text())["lora_file"]["url"] if path.is_file() else spec
    return [{"path": url, "scale": scale}]


async def stage(args, style: str, ideas: list[tuple[str, str]]) -> list:
    """Run every idea through the real upsampler, in the DASHWORLD style."""
    upsampler = PromptUpsampler(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-5"),
        style=style,
        max_chunks=args.max_chunks,
    )
    return [
        await upsampler.upsample(
            raw, author=author, source="fal",
            min_seconds=float(MIN_SECONDS), max_seconds=float(MAX_SECONDS),
            max_chunks=args.max_chunks,
        )
        for raw, author in ideas
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idea", action="append", default=[], help="repeatable")
    parser.add_argument("--idle", type=int, default=0, help="use N of the preset's idle prompts")
    parser.add_argument("--preset", default="dashworld")
    parser.add_argument("--lora", help="fal/dash_lora.json, or a .safetensors URL")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--image-url", help="first frame — a DASH reference render")
    parser.add_argument("--resolution", default="768P", choices=list(COST_PER_SECOND))
    parser.add_argument("--max-chunks", type=int, default=1,
                        help="scenes per idea; 1 keeps each idea to one clip")
    parser.add_argument("--out", type=Path, default=Path("fal/out"))
    parser.add_argument("--confirm", action="store_true", help="actually spend money")
    args = parser.parse_args()

    load_dotenv(CLIENT / ".env")
    for key in ("FAL_KEY", "ANTHROPIC_API_KEY"):
        if not os.environ.get(key):
            print(f"{key} is not set.", file=sys.stderr)
            return 1

    preset = config.load_preset(args.preset)
    ideas = [(text, "you") for text in args.idea]
    ideas += [(p, "auto") for p in preset["idle_prompts"][: args.idle]]
    if not ideas:
        print("Nothing to do: pass --idea or --idle N.", file=sys.stderr)
        return 1

    groups = asyncio.run(stage(args, preset["style"], ideas))
    scenes = [(g, s) for g in groups for s in g.scenes]

    rate = COST_PER_SECOND[args.resolution]
    seconds = sum(max(MIN_SECONDS, min(MAX_SECONDS, round(s.seconds))) for _, s in scenes)
    print()
    for group, scene in scenes:
        secs = max(MIN_SECONDS, min(MAX_SECONDS, round(scene.seconds)))
        print(f"[{secs:>2}s] {group.title}")
        print(f"       {scene.prompt[:150]}...")
    print(f"\n{len(scenes)} clips, {seconds}s at {args.resolution}")
    print(f"COST   ${seconds * rate:.2f}  (${rate}/s)")
    if not args.lora:
        print("\nNote: no --lora, so the character is text-described only —")
        print("the very thing the LoRA exists to fix.")
    if not args.confirm:
        print("\nDry run. Re-run with --confirm to generate and be charged.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for index, (group, scene) in enumerate(scenes, 1):
        secs = max(MIN_SECONDS, min(MAX_SECONDS, round(scene.seconds)))
        arguments = {
            "prompt": scene.prompt,
            "duration": secs,
            "resolution": args.resolution,
            "loras": load_lora(args.lora, args.lora_scale) if args.lora else [],
        }
        if args.image_url:
            arguments["image_url"] = args.image_url
        print(f"\n[{index}/{len(scenes)}] {group.title} ({secs}s)...")
        result = fal_client.subscribe(MODEL, arguments=arguments, with_logs=False)
        url = result["video"]["url"]
        destination = args.out / f"{index:02d}_{group.group_id[:8]}.mp4"
        urllib.request.urlretrieve(url, destination)
        print(f"  -> {destination}")

    print(f"\n{len(scenes)} clips in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
