---
name: dash-clip
description: Generate on-model DASH/EK-0 clips on fal — staging through the dashworld preset, first-frame anchoring, the trained LoRA, and the frame-sheet review loop. Use for test clips, promo beats, or judging character fidelity before Reactor spend.
---

# Generating DASHWORLD clips on fal

One command, from the repo root, venv assumed:

```sh
.venv/bin/python fal/generate.py --idea "<the scene>" --max-chunks 1 \
  --image fal/reference/<anchor> --lora fal/dash_lora.json --lora-scale 0.9 --confirm
```

Scenes are staged by the streaming client's own upsampler in the
`dashworld` preset voice, so a clip judged here is what the live stream
would produce. Keys come from `streaming-client/.env` (`FAL_KEY`,
`ANTHROPIC_API_KEY`).

## Money rules — non-negotiable

- **Always run WITHOUT `--confirm` first.** Both scripts print the scenes
  and the exact price, then exit. Show the user the number before spending.
- 768p costs **$0.075 per second of video** (~$0.75 per 10 s clip).
  LoRA training is **$0.01/step** ($20 at the default 2000).
- Ask the user's remaining fal balance when it matters; never assume.

## Choosing the anchor (`--image`)

The first frame drives composition, palette, **and output aspect ratio** —
a 4:3 anchor makes a 4:3 clip. For 16:9 broadcast use a 16:9 anchor.

| Scene | Anchor |
| --- | --- |
| DASH face-forward | `anchor_img_7792_169.jpg` (both eyes open — the default) |
| DASH in the studio | `anchor_img_7790_169.jpg` |
| EK-0 featured | `11.png` (droid beside DASH, native 1080p) |
| Street/motion | `5.jpg` or `7.jpg` (4:3 — re-crop before broadcast use) |

**Inspect any new anchor before using it.** A winking render (`17.png`)
once propagated its closed eye through entire clips, in base and LoRA
renders alike. The model animates what it is handed.

## The review loop

Never judge from the first frame — the anchor dominates it. Sample across
time and compare against `fal/reference/`:

```sh
for frac in 0.05 0.4 0.75 0.95; do
  ffmpeg -v error -ss $(echo "10*$frac" | bc) -i fal/out/<clip>.mp4 \
    -vf scale=320:-1 -frames:v 1 /tmp/f_$frac.jpg -y
done
```

The back half of a clip is where the adapter and text are doing the work.

## When a character comes out wrong

The description is usually the bug. Fix `streaming-client/presets/
dashworld.json` by **describing the render, not the memory of it** — and
name what the character does NOT have. ("Pointed cat-like ears" produced
upright cat ears and an invented antenna; the fix stated ears angle
outward like wings, no antenna, no limbs.) Commit preset fixes; they are
the durable layer.

## LoRA facts

- Adapter record: `fal/dash_lora.json` (URL + trigger `DASH250`;
  generate.py and side_by_side.py read it). Scale 0.9 is the proven value;
  drop toward 0.8 if expressions over-commit.
- The `/lora` endpoint 422s on an empty `loras` list — `generate.py`
  already routes to the base endpoint when no adapter is passed.
- Retraining: `fal/train_lora.py <clips-dir> --trigger DASH250`. Trainer
  wants ≥10 clips (20–50 better), frames %% 17 == 5 (73 min), one aspect
  per run. Stills become clips via zoompan (see git history of
  `fal/clips/` tooling in the scratchpad scripts).
- Fair fidelity tests use `fal/side_by_side.py` — one staged prompt
  rendered twice, base vs adapter.

## What this path is for

fal proves the creative (per-generation billing, no idle burn). The live
channel runs on Reactor (`DEPLOYING.md`) — at 24/7 duty cycle fal is
~10x the cost of a saturated reserved GPU. Do not pitch fal as the
broadcast backend.

Media directories (`fal/reference/`, `fal/clips*/`, `fal/out/`) are
gitignored — DASHWORLD footage, some unreleased, must never reach the
public repo.
