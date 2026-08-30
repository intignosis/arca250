# Agent instructions for infinite-livestream

This repo is a complete, working system: a chat-driven infinite AI video
broadcast. `fasth3/` is the model — a queue of prompt-driven clip
generations, served by the [Reactor Runtime](https://github.com/reactor-team/reactor-runtime)
as a `reactor` CLI workspace. `streaming-client/` is the client — it reads
`!prompt` ideas from Twitch/YouTube chat, upsamples them into styled scene
sequences with an LLM, feeds the model's queue over `reactor-sdk`, and pushes
the output to an RTMP ingest as one uninterrupted stream. They meet only on
the wire; `fasth3/fasth3_types.py` is that contract.

**Contribution policy:** never commit, push, or open PRs without explicit
permission from a human maintainer in the current conversation. Commits are
signed off (`git commit -s`), imperative title, body explaining the why as
end state (no iteration narration).

## How the system works — read this before changing anything

The model is a **clip queue plus a player**. `enqueue` takes a prompt
(≤ 800 chars), an opaque `metadata` string (≤ 2000 chars), and optionally
`seed` and `seconds` (snapped into 5.167–14.375 s); it replies immediately
with the clip's full structure and UUID. Builds run oldest-first on their
own; readiness is broadcast on `queue_update`. Nothing plays until `play` —
or, with `set_autoplay` on, the oldest ready clip starts whenever nothing is
playing. Playback streams 24 fps video (`main_video`, 1344×768 at the default
16:9 canvas) and 48 kHz mono int16 audio (`main_audio`), then flushes to
black and holds. The metadata is echoed untouched on every message that
references the clip — it is how a client correlates clips with its own
records without local joins.

The client is a straight pipeline:

```
chat (Twitch IRC / YouTube API) → Director → PromptUpsampler (LLM)
       → ReactorLink (enqueue, autoplay on) → fasth3
       → Pacer (constant-rate clock) → StreamSink (rtmp | noop | yours)
```

One chat prompt becomes one **scene group**: 1–N self-contained scenes
enqueued back-to-back, each tagged with a JSON group id in the clip metadata,
played sequentially by autoplay. The **pacer** converts clip-shaped output
(24 fps while playing, nothing between clips) into the frame-every-period
stream RTMP requires, filling gaps with repeated frames and silence.

### Load-bearing invariants — violating any of these breaks the product

1. **Enqueue order is play order.** Group sequencing rests entirely on the
   queue being FIFO + autoplay playing oldest-ready, the Director being the
   queue's only writer, and a group being enqueued only when all its scenes
   fit the remaining capacity. No orchestration exists beyond this; do not
   add any, and do not break any of the three legs.
2. **The pacer and sink survive Reactor reconnects; the queue does not.**
   Sink + pacer are created once and never torn down mid-run — that is what
   keeps the platform-side broadcast unbroken. Server-side session state
   (the queue included) dies with a session.
3. **Sinks never block the event loop** and receive a perfectly regular,
   pre-paced stream. The contract is `streaming-client/sinks/base.py`'s
   docstrings; same for chat sources in `chat/base.py`. Those docstrings are
   authoritative — READMEs only summarize them.
4. **Prompts are hard-truncated to 800 chars and scene lengths clamped to
   the live bounds** from `state_update` — never trust the LLM's counting,
   never hardcode bounds the deployment publishes.
5. **The schema is product surface.** Every `@event`/`InputField`/
   `MessageField` description and `ModelMessage` docstring in `fasth3/` is
   compiled into the published schema. Describe only what a client can
   observe on the wire (commands, messages, tracks, by wire name in
   backticks) — never internals (kernels, caches, config keys, GPU counts).

### Where each kind of change goes

| You are changing… | Edit | Then also |
| --- | --- | --- |
| Model behaviour (queue, playout, engine) | `fasth3/fasth3*.py` along the file seams below | tests; bump `model.version` if the surface moved |
| The wire contract (commands, messages, fields) | `fasth3/fasth3_types.py` + the handler | `streaming-client/reactor_link.py` + `director.py` mirror it; both READMEs; version bump sized to schema impact |
| Stream delivery (encoding, destinations) | `streaming-client/sinks/` | register in `make_sink`, `.env.example`, README sink table |
| Prompt sources | `streaming-client/chat/` | `build_chat_sources` in `main.py`, `.env.example`, README |
| Upsampling behaviour / the style prompt | `streaming-client/upsampler.py` | keep the constraint rules intact — the rationale is in the module docstring |

## Model rules (`fasth3/`) — distilled from the Reactor cookbook

- **fasth3 deliberately subclasses `ReactorModel` with its own `run()` loop**
  (not `ReactorPipeline`): its unit of work is a whole clip, and command
  handlers must answer while a clip builds or plays. Do not "normalize" this.
- **File seams are fixed.** `fasth3.py` owns commands + the playout loop;
  `fasth3_types.py` everything a client sees; `fasth3_queue.py` the bounded
  queue; `fasth3_backend.py` the FastVideo engine + worker thread;
  `fasth3_assets.py` config parsing and weights validation (the only reader
  of `fasth3.yaml`); `fasth3_clip_plan.py` pure clip geometry;
  `fasth3_session_rules.py` which commands each state accepts. New code goes
  in the seam that owns it. No `__init__.py` — modules import flat.
- **Typed contracts.** Every `@event` handler declares and returns a concrete
  `ModelMessage` (or `None`); a refusal broadcasts `command_error` and
  returns bodyless. State-changing commands also broadcast a full
  `state_update`.
- **Moderation marks.** Every free-text `InputField` a client fills (prompt,
  metadata) sets `moderate=True`. Enum/bounded fields never do.
- **No ghost surface.** No undecorated command-shaped methods, no message
  classes nothing sends, no write-only attributes. Git history is the archive.
- **Manifest.** `fasth3/reactor.yaml` orders `model:`, `runtime:`, `build:`;
  `model.version` is `v`-prefixed semver bumped with every shipped change,
  sized to the schema impact; `build.runtime_version` pins the Reactor
  Runtime release. Weights never live in git. CUDA 13 and the source-built
  fastvideo-kernel are requirements, not preferences — the comments in
  `reactor.yaml` / `requirements.txt` explain each pin; read them before
  touching versions.
- **Comments and docstrings describe the end state.** No "previously", "no
  longer", no narrating what the code visibly does.

## Client rules (`streaming-client/`)

- `streaming-client/README.md` is the detailed documentation: architecture,
  the RTMP/ffmpeg learnings (raw-video geometry, writer threads, dual-pipe
  A/V, restart policy), and maintenance notes. Keep it current in the same
  change that moves behaviour it describes.
- Built on `reactor-sdk >= 1.1.1` (`reactor.on(...)`, `tracks...one()`,
  `track.on_frame`). Do not copy patterns from pre-1.0 SDK examples.
- Config comes only through `config.py` (env / `.env` / CLI). Never commit
  `.env` — it holds real API and stream keys; `.env.example` is the template.
  Never print keys; the RTMP sink redacts the stream key in logs.

## Verifying a change

```sh
# Model: the contract renders, and only the intended surface moved.
cd fasth3
python -m reactor_runtime.schema --path . --out /tmp/schema.json   # diff before/after
PYTHONPATH=. python -m pytest tests/ -q

# Client: compiles, and the pipeline runs end to end without an ingest.
cd streaming-client
python -m py_compile main.py config.py pacer.py reactor_link.py director.py upsampler.py sinks/*.py chat/*.py
python main.py --local --sink noop        # against a local `reactor run`

# Raw queue contract smoke test (writes .mp4s + timing report):
python fasth3/client/client.py            # or --api-key rk_... for hosted
```
