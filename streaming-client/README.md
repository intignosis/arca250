# fasth3 streaming client

A chat-driven livestream client for the fasth3 clip-queue model. It reads
`!prompt` requests from Twitch and/or YouTube chat, moderates them, upsamples
each one into a styled sequence of one or more video scenes with an LLM,
enqueues those scenes on a served fasth3 model (local `reactor run` or hosted
with an API key), and forwards the model's video+audio output as one
uninterrupted broadcast to a pluggable **sink** — RTMP (Twitch / YouTube
Live / Kick) today, a no-op sink for dry runs, and an interface designed for
LiveKit / SFU sinks tomorrow. While chat is quiet, an **idle filler** keeps
the queue topped up from a curated prompt list so the stream never runs dry.

Built on `reactor-sdk >= 1.1.1` (the current SDK surface: `reactor.on(...)`,
`reactor.tracks...one()`, `track.on_frame`). The pre-1.0 examples in the
py-sdk repo (`@reactor.on_frame`, `reactor.get_status()`) are an older API —
do not copy patterns from them into this client; their hard-won *ffmpeg*
learnings are already baked into `sinks/rtmp.py`.

## Architecture

```
Twitch IRC ─┐
            ├─▶ Director ──▶ Moderator (OpenAI moderations API)
YouTube API ┘      │   └───▶ PromptUpsampler (OpenAI-compatible LLM)
                   │              ▲
idle_prompts.txt ──┘ (filler)     │
                   ▼  enqueue: scene groups, tagged via clip metadata
              ReactorLink ◀──▶ fasth3 (clip queue, autoplay on)
                   │  24 fps video + 48 kHz audio (per clip, black between)
                   ▼
                 Pacer  ──── constant-rate clock: fills gaps with
                   │         repeated frames + silence
                   │ ──── Overlay (queue badge, now playing, coming up)
                   ▼
              StreamSink ──▶ rtmp (ffmpeg) | noop | (yours)
```

| File | Owns |
| --- | --- |
| `main.py` | Wiring and task lifecycle; nothing else. |
| `config.py` | The only reader of `.env` / environment / CLI. |
| `reactor_link.py` | Everything that touches `reactor_sdk`: connect/reconnect loop, media → pacer, `state_update`/`queue_update` mirror, command sending, message fan-out. |
| `director.py` | Chat prompt → moderate → upsample → enqueue scene groups; idle filler + eviction; per-author cooldown; now-playing narration from metadata. |
| `upsampler.py` | The LLM call and the system prompt; scene validation (char cap, length clamp, chunk-count cap). |
| `moderator.py` | The moderations call and the fail-closed policy. |
| `idle_prompts.txt` | The curated filler prompts, one per line. |
| `pacer.py` | The 24 fps metronome between bursty/clip-shaped model output and the sink's need for a frame + audio every period, forever. Hands each outgoing frame to the overlay. |
| `overlay/` | The per-frame decoration interface (`base.py`) and the shipped status overlay (`status.py`). |
| `sinks/` | The output interface (`base.py`), ffmpeg RTMP (`rtmp.py`), no-op (`noop.py`), factory (`__init__.py`). |
| `chat/` | The chat-source interface (`base.py`), anonymous Twitch IRC (`twitch.py`), YouTube Data API poller (`youtube.py`). |

## The model side, in one paragraph

fasth3 (see `../fasth3/fasth3_types.py`, the authoritative client-facing contract) is
a **clip queue**: `enqueue` takes a prompt (≤ 800 chars), an opaque `metadata`
string (≤ 2000 chars, echoed back on every message that references the clip),
and optionally `seconds` (snapped into 5.167–14.375 s) and `seed`. Builds run
through the queue oldest-first; readiness is announced on `queue_update`.
With `set_autoplay` on — this client turns it on after every connect — the
oldest ready clip plays on its own whenever nothing is playing, streaming on
`main_video` (1344×768 @ 24 fps at the default 16:9 canvas) and `main_audio`
(48 kHz mono int16), then flushes to black. The queue is bounded
(`queue_capacity` in `state_update`, default 10) and a full queue refuses
`enqueue` with a `command_error`.

## Scene groups

One chat prompt becomes one **scene group**. The upsampler picks the shape
the idea calls for: a **single scene** of any legal length, or a **chunked
short story** — up to `MAX_CHUNKS` short clips (each near the model's
minimum length) that read as one story with a setup, development, and
payoff. The director enqueues the group back-to-back. Sequential playback
needs no extra orchestration — it falls out of three facts, and breaks if
any is violated:

1. fasth3's queue is strict FIFO and autoplay plays the oldest ready clip, so
   **enqueue order is play order**.
2. The director is the **only writer** to the queue — both its viewer worker
   and its idle filler serialize group enqueues through one lock — so groups
   never interleave.
3. A group is enqueued only once **all** its scenes fit in the remaining
   capacity, so it can't wedge half-in. Viewer groups may make that room by
   evicting filler clips (see "Idle filler" below).

Each scene's `metadata` carries the group tag as JSON:

```json
{"group_id": "9f2c4e81a0b3", "title": "Neon Alley", "scene": 2, "scenes": 3,
 "author": "viewer_42", "source": "twitch", "generated": false,
 "raw_prompt": "a neon alley..."}
```

The model never reads metadata; it echoes it back on `clip_queued`,
`queue_update`, `clip_started`, `clip_finished`, `clip_failed`, ... — which is
why the director can narrate "scene 2/3 of *Neon Alley*" from a
`clip_started` alone, with no local join. Anything downstream (an overlay, a
chat bot announcing scenes) should be built the same way: read the metadata
echo, not client-side state that a reconnect can lose.

## Prompt upsampling

`upsampler.py` calls one OpenAI-compatible endpoint (`OPENAI_BASE_URL` +
`OPENAI_API_KEY`, so a proxy / vLLM / OpenRouter all work) with a system
prompt that embeds your **style/character** (`STYLE` or `STYLE_FILE`). The
prompt's rules mirror how fasth3 actually behaves — keep them intact when
editing (rationale in the module docstring):

- every scene is an independent clip with **no memory**, so every scene
  prompt must re-describe the full setting/subjects/style from scratch;
- the LLM is asked for < 750 chars but `_sanitize` **hard-truncates to 800**
  (`MAX_PROMPT_CHARS`, the model's server-side cap) because LLMs can't count;
- the scene-prompt **format is distilled from the checkpoint's official
  paper prompts**: `[Shot N]` segments with cut timestamps, `S1`/`S2`
  character tags, a dedicated camera sentence (movement + amplitude +
  speed), dialogue inside the `<d>[Language] ...</d>` speech marker with
  the voice described, explicit constraint assertions ("no readable signs,
  captions, or logos"), and a closing diegetic soundscape;
- **a single-scene generation always runs the maximum clip length**
  (enforced in code, not just asked of the LLM); short lengths are reserved
  for transition chunks inside multi-scene stories, and `seconds` are
  always clamped to the live bounds from `state_update`;
- safety is the moderator's job, upstream — the upsampler stages the idea
  faithfully and never softens or reinterprets it.

Any LLM failure degrades to a single scene made of style + the raw prompt —
the stream never stalls on the upsampler.

## Moderation

Viewer prompts pass the OpenAI moderations API (`moderator.py`) **before**
they reach the upsampler. Moderation deliberately has its own endpoint and
key (`MODERATION_API_KEY` / `MODERATION_BASE_URL`, falling back to the
`OPENAI_*` values): inference gateways often do not expose `/moderations` —
Reactor's corp gateway answers it with `provider_not_allowed` — so moderation
typically points at api.openai.com while upsampling goes through a gateway.

The policy is **fail closed**: a prompt that cannot be checked is rejected,
so a broken moderation endpoint never silently turns moderation off. To run
without moderation, set `MODERATION_ENABLED=0` explicitly — main.py then
warns at startup. Only the raw viewer prompt is checked, and it is the
**only** safety gate: the upsampler deliberately stages ideas faithfully
rather than softening them, so what passes moderation is what gets rendered.
The idle list is curated in this repo and skips the check.

## Idle filler

When chat is quiet, the director's `run_idle` task keeps the model's queue
topped up to `IDLE_QUEUE_TARGET` clips (default 6) from `idle_prompts.txt`
(shuffled, then rotated; override with `IDLE_PROMPTS_FILE`). Filler prompts
run through the same upsampler and style but are forced to **one scene per
group** — the finest eviction granularity, and popping one never truncates a
story — and their metadata carries `generated: true`.

Viewers always outrank filler, in three ways:

- the filler stands down whenever a viewer prompt is pending (including one
  that arrived while the filler's LLM call was in flight — the group is
  dropped before enqueueing);
- when a viewer group doesn't fit the remaining capacity, the director
  **evicts** filler clips with `pop` — newest-queued first, so the filler
  closest to playing (likeliest already built) survives and the stream stays
  fed. Only clips tagged `generated: true` are candidates; a playing clip is
  not in the queue, so playback is never cut;
- eviction recognizes fillers purely by the metadata echo, so it also works
  after a client restart that has no memory of enqueueing them.

The target (6) sits deliberately under the queue capacity (10): the gap is
headroom a viewer group can take without any eviction at all.

## Overlay

Every outgoing frame — live, repeated, or black — passes through the overlay
before the sink, so the broadcast carries live status even between clips.
`OVERLAY_ENABLED` (default on) is the only configuration; *which* overlay
runs is a code decision in `main.py`, and building a different one means
implementing `overlay/base.py`'s `Overlay` (its docstring is the contract:
per-frame budget, never mutate the input frame, state via link listeners
only).

The shipped `StreamStatusOverlay` keeps to the frame's edges (small type on
thin translucent plates):

- **top-left, while playing**: `NOW <title> — scene 2/3 · by <author>`, with
  a dimmer `COMING UP <next>` beneath it (when the next clip is the same
  group it reads `COMING UP scene 3/3` instead of repeating the title);
- **top-left, while idle**: `UP NEXT <title> · by <author>` — or, with an
  empty queue, an invitation to type the chat command;
- **top-right**: `QUEUE n/capacity`.

Everything it shows is reconstructed from the wire — the metadata group tags
(title, author, scene numbering) and the link's queue mirror — so it survives
client restarts, credits idle filler clips to `auto`, and degrades untagged
clips (enqueued by some other client) to their prompt text. Rendering is
cached: Pillow rasterizes a panel only when its text changes; the per-frame
cost is numpy alpha blends (~0.6 ms at 1344×768).

## Sinks

`sinks/base.py` is the contract: by the time a sink sees data, the pacer has
already made it a perfectly regular stream — `send_video` once per period with
one fixed-size rgb24 frame, `send_audio` once per period with exactly one
period of int16 samples, gaps already filled. A sink only encodes and
forwards; it must never block the event loop, owns its own recovery, and
reports health via `alive`.

| `SINK=` | Class | Notes |
| --- | --- | --- |
| `rtmp` | `RtmpSink` | ffmpeg → RTMP(S). Twitch, YouTube Live, Kick are all just ingest URLs. |
| `noop` | `NoOpSink` | Discards everything, logs a heartbeat. Full pipeline dry-run. |

To add one (LiveKit, SFU, file recorder): implement `StreamSink`, register it
in `make_sink` (`sinks/__init__.py`), add its config to `.env.example`, and
extend this table.

## Chat sources

`chat/base.py` is the contract: a long-lived `run(on_prompt)` coroutine that
reconnects internally, delivers each message at most once, never replays
backlog from before startup, and strips the command word. Prompts are
messages starting with `CHAT_COMMAND` (default `!prompt`).

- **Twitch** (`TWITCH_CHANNEL`): anonymous read-only IRC (`justinfan` login) —
  no OAuth, no app registration, no token rotation.
- **YouTube** (`YOUTUBE_VIDEO_ID` + `YOUTUBE_API_KEY`): Data API v3 polling at
  the interval the API requests. The video id must be a **live** broadcast
  (it resolves `activeLiveChatId`). Polling costs quota; don't shorten the
  interval.

Both can run at once. Per-author cooldown (`CHAT_COOLDOWN_S`) and a bounded
backlog in the director keep spam from monopolizing the queue.

## Running

```sh
pip install -r requirements.txt   # ffmpeg must be on PATH for SINK=rtmp
cp .env.example .env              # fill in keys, style, sink, chat

# Dry run against a local `reactor run`, throwing frames away:
python main.py --local --sink noop

# Hosted model, streaming to Twitch, prompts from Twitch chat:
#   .env: REACTOR_API_KEY=rk_..., TWITCH_CHANNEL=yourchannel, STYLE=...
python main.py --sink rtmp --rtmp-url rtmp://live.twitch.tv/app/STREAM_KEY

# YouTube: RTMP_URL=rtmp://a.rtmp.youtube.com/live2/KEY plus
#   YOUTUBE_VIDEO_ID + YOUTUBE_API_KEY for chat.
```

Then type `!prompt a lighthouse in a storm` in chat. Expect: an upsampler log
with the scenes, `queued ... scene 1/n` lines, and — after roughly the clip's
own duration of build time per scene — `[now playing]` lines as autoplay runs
the group. With the idle filler on (the default), `[auto]`-tagged clips fill
the queue within the first minute and the stream shows content instead of
black between viewer groups; with it off, black between groups is the model's
contract, not a bug.

## Learnings baked into this client (do not re-learn these)

From the earlier RTMP clients (py-sdk `rtmp_app` / `story_livestream_app`,
which took many iterations to stabilize) and from driving the fasth3 queue:

- **Raw-video geometry is unforgiving.** One frame whose bytes disagree with
  ffmpeg's `-s WxH` (wrong size, or non-C-contiguous `tobytes()` including
  row padding) shifts every following scanline → "TV static". The pacer
  letterboxes odd sizes onto a fixed canvas; the RTMP sink refuses mismatched
  frames outright rather than corrupting the stream.
- **Never write to a pipe from the event loop.** `stdin.write` blocks when
  ffmpeg stalls; a blocked loop starves WebRTC and everything snowballs. Each
  ffmpeg pipe has its own writer thread behind a bounded drop-oldest queue.
- **Feed audio and video in lockstep, on separate pipes.** fasth3 has real
  synchronized audio (`anullsrc` silence is not enough), and starving one
  ffmpeg input while pushing the other is the classic two-pipe deadlock. The
  pacer delivers both every tick.
- **An audio track is mandatory** — YouTube/Twitch won't take video-only FLV.
  The pacer emits silence when the model is idle, so the encoder never runs dry.
- **ffmpeg dies; the broadcast must not.** The sink restarts it lazily with a
  cooldown and a failure cap, keeping the last stderr lines for the log.
- **The sink outlives Reactor reconnects.** Sink + pacer are created once and
  the connection loop runs behind them, so the platform sees one continuous
  stream while the client rebuilds a session.
- **A constant-rate pacer is not optional for this model.** fasth3's output is
  clip-shaped: 24 fps while a clip plays, *nothing* while the queue idles.
  RTMP needs a frame every period forever. The pacer (FIFO-buffered video and
  audio with the same shallow cap, repeats + silence on underflow, drop-oldest
  on overflow) is what converts one into the other — and buffering both media
  types symmetrically is what keeps A/V sync.
- **The queue dies with the session.** fasth3 resets all session state on a
  new session: after a reconnect, clips that were queued but unplayed are
  gone. Chat prompts still waiting in the director survive (they live
  client-side); a group lost mid-flight is lost. Re-enqueueing on reconnect
  is a possible extension — if added, dedupe via the metadata group tag.
- **Refusals are broadcast, not raised.** A refused command answers with a
  bodyless reply and a `command_error` broadcast carrying the reason. Treat
  "reply without `clip`" as refusal and retry with patience (the director
  does).

## Maintenance notes (for agents)

- **Invariants to preserve:** prompt ≤ 800 chars after sanitization; metadata
  JSON well under 2000 chars; scene groups enqueued contiguously and only
  when they fully fit; all enqueues (viewer and filler) serialized through
  the director's one lock; eviction pops only `generated: true` clips;
  moderation fails closed; autoplay re-enabled on every connect; pacer/sink
  never torn down on reconnect; sinks never block the event loop; overlays
  never mutate the pacer's frame and stay within a few ms per compose.
- `../fasth3/fasth3_types.py` is the wire contract. If the model's schema moves
  (new fields, renamed messages), update `reactor_link.py`'s mirror and the
  director's message handling together, and re-check this README's model
  paragraph.
- New sink → `sinks/` + `make_sink` + `.env.example` + the sink table above.
  New chat platform → `chat/` + `build_chat_sources` in `main.py` + docs.
  Keep the interfaces in `base.py` files authoritative — their docstrings are
  the contract, this README only summarizes them.
- There are no tests here yet; the cheap smoke test is
  `python main.py --local --sink noop` against a local `reactor run` (or the
  reference `../fasth3/client/client.py` for the raw queue contract).
