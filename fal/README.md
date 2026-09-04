# The fal path: proving DASH before renting GPUs

fast-h3 renders a character from a *description*. Every clip is independent,
so "DASH" is re-rolled each time from the preset's style block — the jacket,
the hood and the palette hold, the face does not, and none of it is the DASH
in the catalogue. FastVideo's own preview confirms the checkpoint ships
text-to-video only; image and reference conditioning are unreleased.

fal serves **MiniMax H3** — the model FastH3 is distilled from — with the two
things that close the gap: **first-frame conditioning** and **LoRA adapters**,
still with synchronised audio. So the character can be anchored to a real
DASHWORLD render and a LoRA trained on real DASHWORLD footage.

This directory is the cheap way to answer "does he actually look like DASH?"
without Docker, a 148 GB upload, or a reserved 8xB200 pod.

## Cost, and why this is not the way to run a channel

| | Price |
| --- | --- |
| LoRA training | $0.01/step — 2,000 steps = **$20** |
| Generation @ 768p | **$0.075 per second of video** |

Billed per generation, so nothing idles. That is the whole advantage, and
also the limit: at $0.075/s, running continuously is ~$194,000/month against
~$17,000 for a saturated 4xB200 on Reactor. **fal answers the creative
question; Reactor runs the channel.** See [`../DEPLOYING.md`](../DEPLOYING.md).

## Use

```sh
pip install -r fal/requirements.txt
export FAL_KEY=...                      # ANTHROPIC_API_KEY comes from streaming-client/.env

# 1. Train the LoRA on DASHWORLD's own footage — the visualizers and the film
#    already qualify: 10 clips minimum, 20-50 better, each >= 73 frames (~3 s).
python fal/train_lora.py ./clips --trigger DASH250            # dry run: prints the price
python fal/train_lora.py ./clips --trigger DASH250 --confirm  # $20 at the default 2000 steps

# 2. Generate test clips, staged by the same upsampler the stream uses
python fal/generate.py --idle 3                                       # dry run
python fal/generate.py --idle 3 --lora fal/dash_lora.json \
    --image-url https://.../dash-reference.png --confirm
```

Both scripts refuse to spend without `--confirm`; a bare run prints exactly
what it would generate and what it would cost, and exits.

`generate.py` imports `streaming-client/`'s own `PromptUpsampler` and reads
`presets/dashworld.json`, so scenes are staged in the identical voice the
live stream uses — the only thing that changes is where they are rendered.

## Rendering a training set from the rig

`render_training_set.py` renders LoRA training clips from `Dash - rigged.blend`
headless — canonical geometry, native 16:9 at 24 fps, no captions, no
compression artefacts, and free. It builds its own orbit camera and key/rim
lights rather than trusting the file's, sweeps three lighting states, and
emits ~23 clips: turntables, busts, face close-ups, boot passes, low hero
angles, plus expression sweeps. The rig carries the ARKit facial set (55
shapes) but ships no baked animation, so expressions are driven directly —
neutral to shout, smirk, drained, alert and back — while the camera moves.

```sh
blender --background "Dash - rigged.blend" \
    --python fal/render_training_set.py -- --out fal/clips_rig
python fal/train_lora.py fal/clips_rig --trigger DASH250
```

**Blender 4.5 LTS**, not 5.x — 5.0 dropped Intel macOS builds, and the
Homebrew cask installs the Apple Silicon one, which macOS kills on sight on
this machine. Clip lengths are 73 and 124 frames, both satisfying the
trainer's `frames %% 17 == 5` rule, so `auto_scale_input` stays off.

## Notes

- `--max-chunks 1` keeps one clip per idea, which is what you want when
  judging character fidelity. Raise it to see multi-scene continuity.
- fal takes whole-second durations, clamped here to 5-10 s. fast-h3's legal
  range (5.04-14.375 s) does not apply.
- Without `--lora` the character is text-described only — useful as the
  before-picture in a before/after, not as a fidelity test.
