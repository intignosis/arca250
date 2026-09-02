# Deploying fast-h3 to Reactor

What it takes to get `arca250live` on the air, in the order it has to happen,
with the numbers that were actually measured rather than assumed. Written
2026-09-01 against Reactor beta; re-check the live rates before spending.

## Where things stand

| | State |
| --- | --- |
| Streaming client | Works. Preset loads, chat joins `#arca250live`, Claude upsampling verified against the live API. |
| `ANTHROPIC_API_KEY` / `MODERATION_API_KEY` | Set in `streaming-client/.env`. |
| `REACTOR_API_KEY` | Set, valid — the account authenticates against `api.reactor.inc`. |
| **fast-h3 on Reactor** | **Not deployed.** The account owns no models; there is nothing for the client to connect to. |

Everything below exists to close that last row.

## What the build host needs

The CLI docs are explicit: "your host needs just two things: the CLI, and a
running Docker daemon." The Python runtime and media libraries ship inside
the image the CLI builds. **No GPU is needed to build** — `reactor.yaml`
sets `TORCH_CUDA_ARCH_LIST: "10.0a;10.3a"` precisely so `nvcc` cross-compiles
the Blackwell kernels on a machine that has no Blackwell in it.

This Mac qualifies, and it is worth noting why, because renting a Linux box
is the obvious-but-unnecessary move:

- `x86_64` — the image targets `linux/amd64`, so Docker builds it natively
  with no QEMU emulation penalty. (An Apple Silicon Mac would emulate, and
  a source build of CUDA kernels under QEMU is measured in days.)
- 16 cores — `reactor.yaml` asks for `MAX_JOBS: 32`; it will use what it has.
  Expect the kernel compile to take **1-2 hours** rather than the ~30 minutes
  its authors see on a 32-core box.
- 1.2 TB free — enough for the image plus the 148 GB weights bundle.

Missing: **Docker Desktop**. Install it and give its VM ~16 GB of the 32 GB
and at least 250 GB of disk.

What this machine cannot do is `reactor run`, which serves the model locally
and needs real Blackwell silicon. There is no local smoke test. The first
time this model runs, it runs on Reactor, and it bills.

## The sequence

```sh
# 1. CLI (Homebrew is already installed here)
brew install reactor-team/tools/reactor-cli
reactor version
reactor auth login
reactor doctor                      # confirms Docker is reachable

# 2. Weights: ~148 GB from Hugging Face into the workspace
#    The bundle layout fast-h3 expects is in fast-h3/README.md.
huggingface-cli download \
  FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree \
  --local-dir fast-h3/weights/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree

# 3. Build the container. Long. Compiles fastvideo-kernel and flash-attn-4
#    from source; see the pins and their rationale in fast-h3/requirements.txt.
cd fast-h3
reactor validate
reactor build

# 4. Register and ship, weights included
reactor model register
reactor model deploy
reactor model status
```

Then put the deployed model's slug in `streaming-client/.env` as
`REACTOR_MODEL`, drop `--local`, and the client connects to it.

## Cost, from Reactor's live pricing API

`GET api.reactor.inc/pricing` returns real rates. 10,000 credits = $1, billed
**per second from the moment a session reaches `ready`** — idle and streaming
bill identically.

| Reference model | GPUs | Credits/sec | $/hour |
| --- | --- | --- | --- |
| `visko-orbis-dynamic` | 8xB200 | 209 | **$75.24** |
| `visko-orbis-stable` | 7xB200 | 97 | $34.92 |
| `lingbot`, `lingbot-world-2` | 4xB200 | 33 | $11.88 |
| `helios`, `longlive-v2`, `x2` | 1xB200 | 17 | $6.12 |

Rates are per model, not per card — `lingbot` is $2.97/GPU-hr and
`visko-orbis-dynamic` $9.40 — so a custom deployment's rate is set at
registration, not derived from `gpu.count`. Treat `visko-orbis-dynamic` as
the ceiling for an 8-GPU fast-h3 and confirm the real number before running.

**Free:** `connecting` and `waiting`. Model load and the warm-up builds happen
there, so the ~10 minute boot costs nothing. Image builds are local and
unbilled. **Billed:** every wall-clock second after `ready`, including a
recoverable disconnect, because the GPU stays reserved.

Two consequences worth acting on:

1. **Keep `warmup_lengths: "all"`.** Warm-up is free; a mid-session compile
   stall on an unwarmed clip length is ~20 s of billed time. Warm everything
   up front, pay for nothing.
2. **Never leave a session open.** An idle session bills at the full rate.

### Register on ONE GPU first

FastVideo's own preview post puts the checkpoint's minimum at **1x B200**, and
the memory arithmetic in `fasth3.yaml` agrees: the resident footprint is
~92 GB per rank against 183 GB on a B200, so a single card holds the
replicated transformer and its text encoder with no offloading config change
at all. `gpu.count: 1` also satisfies the "must divide H3's 56 attention
heads" rule.

What GPU count buys is **latency, not clips per dollar**. Scaling is poor at
the top — 4 to 8 GPUs takes a 15 s clip from ~15.5 s to ~12.9 s, 20% faster
for twice the money — which means cost per clip stays roughly flat while the
burn rate per second does not:

| | ~Build/clip | Reference $/hr | ~$/clip | Clips for $10 | $10 lasts |
| --- | --- | --- | --- | --- | --- |
| 1xB200 | ~50 s (est.) | $6.12 | ~$0.085 | ~118 | **~100 min** |
| 4xB200 | 14.4 s (measured) | ~$24 | ~$0.098 | ~102 | ~25 min |
| 8xB200 | 12.9 s (measured) | $75.24 | ~$0.270 | ~37 | ~8 min |

Same clips either way; wildly different wall-clock. On one GPU the $10 lasts
an hour and a half instead of eight minutes — room to watch the output, fix a
prompt, and re-run, instead of racing a meter.

**The rule that decides the count later:** a live feed needs build time under
clip duration, or every clip boundary waits on the GPUs. At 14.4 s per
14.375 s clip, **four is the realtime floor**. One GPU at ~0.3x realtime
renders fine test clips and could never sustain a broadcast. So: one GPU to
judge the creative work, four when `arca250live` goes live, eight only once
continuous running shows boundaries stalling.

### An unresolved timing discrepancy

FastVideo's post reports **47.2 s** for a 15 s clip on 8xB200. `fast-h3/README.md`
reports **14.4 s** on four. Both cannot describe the same code path.

The likely reconciliation is in `fast-h3/README.md` itself: the published
`fastvideo-kernel` wheel's `sm_100a` binary "fails *every* launch on this
driver", and the Triton fallback route is "~2.5x slower". 14.4 x 2.5 = 36 s,
the right order of magnitude for 47.2 s. That would make the blog's figure the
un-tuned route and the repo's the source-built sm100a profile — which is
exactly why `reactor.yaml` compiles the kernel from source.

Worth confirming on the first deployment, because it is load-bearing: at
47 s/clip nothing about a realtime feed works at any GPU count.

### Register on four GPUs when going live

For a credit-limited first run, four B200s beat eight. Eight build a 15 s clip
in ~12.9 s against ~15.5 s on four — 20% faster for 100% more money:

Move to eight only when clip boundaries start waiting on the GPUs. `gpu.count`
lives in `fast-h3/reactor.yaml`.

## Running continuously, when it gets there

At the 8xB200 reference rate, 24/7 is **~$54,000/month**. A four-hour nightly
show at the same rate is ~$9,000/month; on four GPUs, ~$1,400/month.

The Claude layer is lunch money by comparison: ~$0.04 per viewer prompt and
~$4.50/hour of idle filler on `claude-opus-5`, roughly 5x less on
`claude-haiku-4-5` (`ANTHROPIC_MODEL` in `.env`). Moderation is negligible —
it only fires on viewer prompts.

## The cheaper first move: prove DASH on fal

Everything above buys a *channel*. None of it answers whether the character
looks like DASH — and on this checkpoint it cannot, because fast-h3 renders
from a description and re-rolls the character every clip.

fal serves **MiniMax H3**, the model FastH3 is distilled from, with
first-frame conditioning and LoRA adapters, audio included. That is the
fidelity answer, and it needs no Docker, no 148 GB upload, and no reserved
pod. See [`fal/README.md`](fal/README.md); the scripts are wired and refuse
to spend without `--confirm`.

| Route | Setup | Per second of video | 24/7 month |
| --- | --- | --- | --- |
| fal MiniMax H3 @ 768p | none — an API key | $0.075 | ~$194,000 |
| Reactor 4xB200 fast-h3 | build + 148 GB + register | ~$0.007 | ~$17,000 |
| Reactor 8xB200 fast-h3 | build + 148 GB + register | ~$0.019 | ~$54,000 |

The structure inverts on duty cycle: fal has no idle burn and no floor, so it
wins for generating a little; a saturated reserved GPU is 4-11x cheaper per
video-second, so Reactor wins for running continuously. **Use fal to decide
whether the creative works, Reactor to run it once it does.**

The first spend worth making is ~$20 of LoRA training on DASHWORLD's existing
visualizers plus ~$15 of test clips — which settles the character question
before a single GPU-hour is reserved.

## What the checkpoint cannot do

FastVideo's post is explicit: text-to-video-and-audio is the shipped task.
**Image-to-video is not supported**; first/last-frame (FL2VA) and
reference-to-video are "in development". This settles the `zhuma/fasth3-i2v`
branch question — it anchors on conditioning the checkpoint does not have.
Build on `main`. Quantization (FP8, NVFP4) and an 8-step variant are also
unreleased, so today's cost is the cost.

## Open questions for Reactor

1. Is weights-bundle **storage** billed, and at what rate? 148 GB sits there
   whether or not a session runs.
2. What rate will a custom 8xB200 model be assigned, and who sets it?
3. Promotional credits expire in **90 days** — when does the current $10
   lapse?
