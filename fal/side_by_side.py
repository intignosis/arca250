"""One staged scene, rendered twice: base H3 vs the DASH LoRA.

The only fair fidelity test: identical prompt, identical first frame,
identical duration — the adapter is the sole variable. Output lands in
fal/out/sbs/ as base.mp4 and lora.mp4.

    python fal/side_by_side.py --idea "dash at the studio console" --confirm
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
load_dotenv(CLIENT / ".env")

import config  # noqa: E402
from upsampler import PromptUpsampler  # noqa: E402

BASE, LORA_EP = "minimax/h3/image-to-video", "minimax/h3/image-to-video/lora"
RATE = 0.075  # $/s at 768P


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idea", default="DASH alone at his studio console at night, turning to face the camera as the beat comes together")
    parser.add_argument("--image", type=Path, default=Path("fal/reference/17.png"))
    parser.add_argument("--lora", type=Path, default=Path("fal/dash_lora.json"))
    parser.add_argument("--seconds", type=int, default=8)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    for key in ("FAL_KEY", "ANTHROPIC_API_KEY"):
        if not os.environ.get(key):
            print(f"{key} is not set.", file=sys.stderr); return 1
    if not args.lora.is_file():
        print(f"LoRA record not found: {args.lora} — has training finished?", file=sys.stderr)
        return 1
    record = json.loads(args.lora.read_text())
    lora_url = record["lora_file"]["url"]
    trigger = record.get("trigger_phrase", "")

    preset = config.load_preset("dashworld")
    upsampler = PromptUpsampler(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-5"),
        style=preset["style"], max_chunks=1,
    )
    group = asyncio.run(upsampler.upsample(
        args.idea, author="sbs", source="fal",
        min_seconds=float(args.seconds), max_seconds=float(args.seconds), max_chunks=1))
    prompt = group.scenes[0].prompt
    # The trigger phrase is what activates the adapter; prepend it for both
    # renders so the text is byte-identical and only the weights differ.
    if trigger and trigger not in prompt:
        prompt = f"{trigger}. {prompt}"

    print(f"scene: {group.title}\nprompt ({len(prompt)} chars): {prompt[:180]}...")
    print(f"anchor: {args.image}  duration: {args.seconds}s")
    print(f"COST   ${2*args.seconds*RATE:.2f}  (two renders)")
    if not args.confirm:
        print("\nDry run. Re-run with --confirm."); return 0

    image_url = fal_client.upload_file(args.image)
    out = Path("fal/out/sbs"); out.mkdir(parents=True, exist_ok=True)
    for label, endpoint, extra in (
        ("base", BASE, {}),
        ("lora", LORA_EP, {"loras": [{"path": lora_url, "scale": 1.0}]}),
    ):
        print(f"\n[{label}] rendering...")
        result = fal_client.subscribe(endpoint, arguments={
            "prompt": prompt, "duration": args.seconds,
            "resolution": "768P", "image_url": image_url, **extra,
        }, with_logs=False)
        dest = out / f"{label}.mp4"
        urllib.request.urlretrieve(result["video"]["url"], dest)
        print(f"  -> {dest}")
    print("\ndone: fal/out/sbs/base.mp4 vs fal/out/sbs/lora.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
