---
name: reactor-streaming-client
description: Full context for the streaming client in this repo — the philosophy that splits it from the fasth3 model, the chat→moderation→upsampling→two-queue→playout pipeline, every load-bearing scheduling policy (viewer priority, FIFO, drops, backpressure), the media path (pacer, sinks, overlay), and how to run and verify it. Read before changing anything under streaming-client/, debugging the broadcast, adding a sink or chat platform, or tuning the prompt/scheduling behaviour.
---

# The streaming client: full working context

`streaming-client/` turns a served fasth3 model into an **unattended,
chat-programmed, never-ending broadcast**: viewers type `!prompt <idea>` in
Twitch or YouTube chat, an LLM stages each idea into scenes, the model
renders them, and the output streams to an RTMP ingest with a live status
overlay. This file is the context a new agent needs; per-module detail lives
in [`streaming-client/README.md`](../../streaming-client/README.md) and the
code's own docstrings, and the model's side of the story in the
`reactor-fasth3-model` skill.

## 1. The philosophy: the model renders, the client decides

The one design rule everything else hangs off:

- **fasth3 is a renderer with two queues and zero policy.** It builds the
  generation queue front-first (always, on its own), parks built clips in
  the playout queue, and plays exactly what it is told. It never knows who
  asked for a clip or why one outranks another.
- **This client is all policy.** What builds next (`enqueue`'s `position`),
  what plays next (`play {clip_id}`), what gets dropped (`pop`), and what
  every clip *means* — viewer request vs idle filler, scene 2 of 3, who
  asked — live here, carried exclusively in the **metadata echo**: an opaque
  JSON tag the client writes at enqueue time and the model returns untouched
  on every message that references the clip (`group_tag.py` owns the
  format).

Consequences an agent must not undo: the client reconstructs *everything*
from the wire (state mirrors + metadata echo), so it survives its own
restarts and never keeps scheduling state the queues do not confirm; and no
client concept (viewer, filler, group, author) may leak into the model —
if a change needs the model to know one, the design is wrong.

## 2. The pipeline

```
Twitch IRC ─┐                                       idle_prompts.txt
            ├─▶ Director ─▶ Moderator ─▶ PromptUpsampler ◀── (filler, quiet
YouTube API ┘      │            (LLM, fail-closed)            chat only)
                   ▼ enqueue(position) / play / pop — scene groups tagged in metadata
              ReactorLink ◀──▶ fasth3: [generation q] ─build─▶ [playout q]
                   │  24 fps video + 48 kHz mono audio (per clip, black between)
                   ▼
                 Pacer ── constant 24 fps clock; fills gaps with repeats+silence
                   │  └── Overlay (READY n/cap · GEN m; NOW/COMING UP/UP NEXT)
                   ▼
              StreamSink ─▶ rtmp (ffmpeg → Twitch/YouTube/Kick) | noop | yours
```

| Module | Owns |
| --- | --- |
| `main.py` | Wiring and task lifecycle only. Tasks: link, director, playout, idle filler, chat sources, pacer. |
| `config.py` | Sole reader of `.env`/environment/CLI. |
| `reactor_link.py` | Everything `reactor_sdk`: connect/reconnect loop, both queue mirrors + `state_update` mirror, command sending, message fan-out, media → pacer, local orphan clearing. |
| `director.py` | All scheduling: intake (cooldown, budget drop), moderate → upsample → positional enqueue; `run_playout` (the only sender of `play`); `run_idle` (filler); backpressure relief. |
| `group_tag.py` | The metadata tag format + the shared policies: `pick_next` (play order), `viewer_insert_position` (build order), `is_generated`. |
| `upsampler.py` | The LLM call, the system prompt (distilled from fasth3's official paper prompts), scene validation, retries. |
| `moderator.py` | OpenAI moderations on its own endpoint; fail-closed. |
| `pacer.py` | The 24 fps metronome between clip-shaped model output and the sink. |
| `overlay/` | Per-frame status drawing; contract in `base.py`, shipped overlay in `status.py`. |
| `sinks/` | Delivery; contract in `base.py`, ffmpeg RTMP in `rtmp.py`, `noop.py` for dry runs. |
| `chat/` | Prompt sources; contract in `base.py`, anonymous Twitch IRC, YouTube Data API poller. |

## 3. The scheduling policies (all load-bearing)

A clip is **viewer content** unless its metadata tag says `generated: true`
(untagged clips from other clients count as viewer content). On that single
bit rests:

1. **Build priority with viewer FIFO.** A viewer group enters the generation
   queue at `viewer_insert_position`: ahead of every waiting filler, behind
   every waiting viewer clip — first-come-first-served among viewers, filler
   slides back, nothing popped or wasted. Filler appends.
2. **Group contiguity.** The director is the queues' only writer; its viewer
   worker and idle filler serialize whole-group enqueues through one lock,
   and a group is enqueued only when all its scenes fit the generation queue
   (waiting filler is evicted to make room). Consecutive positions keep a
   group's scenes adjacent, so build order = scene order.
3. **Play priority.** `run_playout` is the only sender of `play` (autoplay
   stays off) and always chooses via `pick_next` over the playout queue:
   first viewer clip in queue order, else the front filler. The overlay's
   "coming up" reads the same function, so what is announced is what plays.
4. **Backpressure relief.** Generation pauses while the playout queue is
   full. When what fills it is built filler and a viewer clip waits to
   build, the playout loop pops one filler per tick (newest first) until the
   build resumes — one per tick so a draining queue gets the chance to make
   room by playing instead.
5. **Drops over stalls.** When the viewer backlog across both queues reaches
   the deployment's clip budget (`playout_capacity`), new chat prompts are
   dropped *before* costing a moderation or LLM call — one prompt never
   stalls the pipeline. Every capacity is read live from `state_update`,
   never assumed.
6. **The idle filler stands down** whenever viewer work is pending, gates on
   `IDLE_QUEUE_TARGET` across both queues (self-clamped under the live
   playout capacity — see below), and always produces single-scene
   max-length groups — the finest pop granularity, and popping one never
   truncates a story.

### What the two capacities mean to this client

The deployment publishes both queue bounds live in `state_update`, and the
client adapts to whatever they are — **no constant in this codebase may
encode a capacity**, because a deployment can resize either knob without the
client changing:

- **`playout_capacity`** (the model's `queue_size`) is the scarce resource —
  built clips in host RAM — so the client treats it as *the clip budget*:
  the viewer backlog across both queues drops new prompts at this number
  (policy 5), the backpressure rule watches the playout queue against it
  (policy 4), and the idle target self-clamps to at least one slot below it,
  since filler must never be what fills the playout queue to the brim
  (a full playout queue pauses builds; the headroom is where the next viewer
  clip lands).
- **`generation_capacity`** is cheap backlog (prompts, not pixels): the
  client only checks that a whole group fits before enqueueing, evicting
  waiting filler if needed.
- **A mismatch between the two is normal, not an error.** A big generation
  queue over a small playout queue just means a long build backlog draining
  through few slots — every policy above keeps working, only waits grow. A
  tiny deployment (say `queue_size: 3`) shrinks the clip budget, the idle
  target, and the drop threshold in one motion, with no configuration
  change here.

## 4. The media path

- **The pacer is not optional.** fasth3 emits 24 fps *while a clip plays*
  and nothing between clips; a live sink needs a frame and audio every
  period forever. The pacer buffers video and audio symmetrically (that
  symmetry is A/V sync), fills gaps with repeated frames + silence, and —
  critically — **outlives Reactor reconnects** together with the sink, so
  the platform sees one unbroken broadcast while sessions churn.
- **Sinks and overlays are contracts, not conventions.** By the time a sink
  sees data the stream is perfectly regular; a sink only encodes and
  forwards, never blocks the event loop, and owns its own recovery
  (`sinks/base.py`). An overlay composes per tick on a copy, never mutates
  the pacer's frame, pre-renders text into cached rasters (~0.6 ms/frame),
  and gets state only by listening (`overlay/base.py`). The docstrings in
  both `base.py` files are the authoritative contracts.
- **The RTMP sink carries hard-won ffmpeg learnings** (exact raw-video
  geometry or "TV static", per-pipe writer threads, dual-pipe A/V fed in
  lockstep, mandatory audio track, lazy restarts with a failure cap) — the
  README's "Learnings" section lists them; do not relearn them.

## 5. The LLM parts

- **Upsampler** (`upsampler.py`): one OpenAI-compatible call per idea
  (`OPENAI_BASE_URL` — a corp gateway works). The system prompt embeds the
  configured `STYLE` (an original cartoon-sitcom world with a crossover
  cast) and is **distilled from fasth3's official paper prompts**: `[Shot N]`
  segments with cut timestamps, `S1`/`S2` character tags, a camera sentence
  with amplitude and speed, dialogue inside `<d>[Language] ...</d>` with the
  voice described, explicit constraint assertions, closing soundscape. Keep
  its constraint rules intact — each traces to measured model behaviour
  (rationale in the module docstring). Length policy: a single-scene
  generation always runs the maximum clip length (enforced in code); short
  lengths are transition chunks inside stories only. Up to 3 attempts per
  idea, each request-tagged (the gateway caches identical requests), then a
  raw-prompt fallback; over-long output is cut at a sentence boundary.
- **Moderator** (`moderator.py`): the *only* safety gate — the upsampler
  deliberately stages ideas faithfully. Own endpoint/key (`MODERATION_*`),
  because inference gateways typically do not expose `/moderations`;
  fail-closed on errors; disabled only by explicit `MODERATION_ENABLED=0`
  (the startup log calls it out). Idle prompts are curated and skip it.

## 6. Resilience facts

- **Sessions die; the broadcast does not.** The link rebuilds sessions in a
  loop behind the pacer/sink. Server-side session state (both queues) dies
  with a session; chat prompts still in the director survive client-side.
- **A dead client leaves the local runtime "orphaned"** (409 on connect for
  a minute+). The link force-clears it with a bare `POST /stop_session` —
  local mode only, only on the `orphaned` refusal, never on `streaming`
  (that may be someone else's session). Teardown does the same when its own
  disconnect fails. Measured: reconnects went from ~90 s to ~7 s.
- **Refusals are broadcast, not raised**: a refused command answers
  bodyless and `command_error` carries the reason; the director treats
  "reply without `clip`" as refusal and retries with patience.
- Frame handlers register **by wire name before connect** (`main_video`,
  `main_audio`); querying the track list after connect races the session's
  track declaration.

## 7. Running and verifying

```sh
cd streaming-client
pip install -r requirements.txt      # ffmpeg on PATH for SINK=rtmp
cp .env.example .env                 # keys, style, sink, chat channel
python main.py --local --sink noop   # full pipeline, delivery discarded
python main.py                       # everything from .env
```

- `.env` holds real keys (gateway, Twitch stream key) and is gitignored —
  never commit or print it. `REACTOR_LOCAL_URL` points local mode at a
  runtime on any port.
- No test suite here; the checks that exist: `python -m py_compile` over the
  modules, the policy smoke tests exercised in review (fake link, no GPU),
  and the real smoke: `--local --sink noop` against a `reactor run`, then
  type the chat command and watch the `[director]` / `[now playing]` lines.
- The client requires the model's **v0.5.0 two-queue contract**
  (`clip_generated`, `position`, both-queue `queue_update`); against an
  older image every enqueue still works but the mirrors stay empty — if the
  queues log as 0/0 while builds clearly run, the served image predates the
  contract.

## 8. Keeping this skill true

This file is the context handoff between agents, the same way
`reactor-fasth3-model` is for the model. When work under `streaming-client/`
changes the pipeline, a policy in section 3, a contract in the `base.py`
files, or the run/verify story, update this skill **in the same change** —
a stale skill poisons the next session's assumptions.
