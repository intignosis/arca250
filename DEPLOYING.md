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

### Register on four GPUs first

For a credit-limited first run, four B200s beat eight. Eight build a 15 s clip
in ~12.9 s against ~15.5 s on four — 20% faster for 100% more money:

| | Build/clip | $/sec | Cost per clip | Clips for $10 |
| --- | --- | --- | --- | --- |
| 8xB200 | 12.9 s | 2x | 25.8 units | ~28 |
| 4xB200 | 15.5 s | 1x | 15.5 units | **~47** |

Eight GPUs buy latency headroom for a feed where builds must outpace playback
24/7. They do not buy throughput per dollar. Change `gpu.count` to 4 in
`fast-h3/reactor.yaml` for the first run; move to 8 when the channel is
running continuously and clip boundaries start waiting on the GPUs.

## Running continuously, when it gets there

At the 8xB200 reference rate, 24/7 is **~$54,000/month**. A four-hour nightly
show at the same rate is ~$9,000/month; on four GPUs, ~$1,400/month.

The Claude layer is lunch money by comparison: ~$0.04 per viewer prompt and
~$4.50/hour of idle filler on `claude-opus-5`, roughly 5x less on
`claude-haiku-4-5` (`ANTHROPIC_MODEL` in `.env`). Moderation is negligible —
it only fires on viewer prompts.

## Open questions for Reactor

1. Is weights-bundle **storage** billed, and at what rate? 148 GB sits there
   whether or not a session runs.
2. What rate will a custom 8xB200 model be assigned, and who sets it?
3. Promotional credits expire in **90 days** — when does the current $10
   lapse?
