"""Train a DASH LoRA on fal, from DASHWORLD's own footage.

Why this exists: fast-h3 is text-to-video only, so the character it draws is
whatever the style block describes — a DASH-like figure, re-rolled every
clip, never the DASH in the catalogue. fal serves MiniMax H3 (the model
FastH3 is distilled from) with first-frame conditioning *and* LoRA adapters,
so a LoRA trained on real DASHWORLD clips is what makes the character
himself rather than a description of himself.

The trainer wants at least 10 clips and prefers 20-50; each needs at least
73 frames (~3 s at 24 fps). The visualizers and the film already qualify.

    python fal/train_lora.py ./clips --trigger "DASH250" --steps 2000

Costs $0.01 per step. Nothing is submitted without --confirm; a bare run
prints the plan and the price and exits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import fal_client

TRAINER = "minimax/h3/i2v/trainer"
COST_PER_STEP = 0.01
CLIP_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv")
# The trainer's frame count must satisfy frames % 17 == 5; 73 is its default
# and the shortest clip it will take without auto_scale_input.
MIN_FRAMES = 73


def collect_clips(source: Path) -> list[Path]:
    """Every trainable clip under `source`, or the single file it names."""
    if source.is_file():
        return [source]
    return sorted(
        p for p in source.rglob("*") if p.suffix.lower() in CLIP_SUFFIXES
    )


def pack(clips: list[Path], into: Path) -> Path:
    """Zip the clips flat — the trainer reads names, not directory structure."""
    with zipfile.ZipFile(into, "w", zipfile.ZIP_STORED) as archive:
        for clip in clips:
            archive.write(clip, arcname=clip.name)
    return into


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="directory of clips, or a .zip")
    parser.add_argument("--trigger", default="DASH250", help="trigger phrase")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--rank", type=int, default=32, choices=[8, 16, 32, 64, 128])
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--frames", type=int, default=MIN_FRAMES)
    parser.add_argument(
        "--out", type=Path, default=Path("fal/dash_lora.json"),
        help="where the trained LoRA's URLs are written",
    )
    parser.add_argument("--confirm", action="store_true", help="actually spend money")
    args = parser.parse_args()

    if not os.environ.get("FAL_KEY"):
        print("FAL_KEY is not set.", file=sys.stderr)
        return 1
    if args.frames % 17 != 5:
        print(f"--frames must satisfy frames %% 17 == 5; {args.frames} does not.", file=sys.stderr)
        return 1

    if args.source.suffix.lower() == ".zip":
        archive, clips = args.source, None
    else:
        clips = collect_clips(args.source)
        if len(clips) < 10:
            print(
                f"found {len(clips)} clips under {args.source}; the trainer wants at "
                "least 10, and 20-50 trains a more robust concept.",
                file=sys.stderr,
            )
            return 1
        archive = None

    cost = args.steps * COST_PER_STEP
    print(f"trainer        {TRAINER}")
    print(f"source         {args.source}")
    if clips is not None:
        print(f"clips          {len(clips)}")
    print(f"trigger        {args.trigger!r}")
    print(f"steps / rank   {args.steps} / {args.rank}")
    print(f"COST           ${cost:.2f}  (${COST_PER_STEP:.2f}/step)")
    if not args.confirm:
        print("\nDry run. Re-run with --confirm to submit and be charged.")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        if archive is None:
            archive = pack(clips, Path(tmp) / "dataset.zip")
            print(f"packed {len(clips)} clips -> {archive.stat().st_size / 1e6:.0f} MB")
        print("uploading dataset...")
        dataset_url = fal_client.upload_file(archive)

        print("training (this is long; progress streams below)")
        result = fal_client.subscribe(
            TRAINER,
            arguments={
                "training_data_url": dataset_url,
                "trigger_phrase": args.trigger,
                "number_of_steps": args.steps,
                "rank": args.rank,
                "learning_rate": args.learning_rate,
                "number_of_frames": args.frames,
                "aspect_ratio": args.aspect_ratio,
                "frame_rate": 24,
            },
            with_logs=True,
            on_queue_update=lambda status: [
                print(f"  {entry['message']}")
                for entry in getattr(status, "logs", None) or []
            ],
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"trigger_phrase": args.trigger, **result}, indent=2) + "\n"
    )
    print(f"\nLoRA: {result['lora_file']['url']}")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
