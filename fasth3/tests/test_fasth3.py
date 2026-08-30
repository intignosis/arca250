"""FastH3's clip geometry, queue, command contract, playout loop, schema and manifest.

Everything here runs on a laptop: the GPU work sits behind the backend, which
these tests replace with a fake that builds instantly, so the real queueing,
ordering, pacing, emission and teardown all run.

Run from the model folder: ``PYTHONPATH=. python -m pytest tests/ -q``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

import fasth3_clip_plan as clip_plan
import fasth3_session_rules as session_rules
from fasth3 import EMIT_FRAMES, FastH3
from fasth3_assets import FastH3Config, load_config
from fasth3_backend import ClipJob
from fasth3_queue import ClipQueue, new_entry
from fasth3_types import ClipInfo

MODEL_DIR = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------- clip geometry
#
# The rules a clip length has to satisfy to be generatable.


def test_bounds_are_the_generatable_range():
    """5 s aligns up to 124 frames; 15 s aligns up to 362 and is out of range."""
    assert clip_plan.MIN_FRAMES == 124
    assert clip_plan.MAX_FRAMES == 345
    # The ceiling is not 15 s: 360 frames aligns up to 362, which is 15.083 s
    # and past the cap, so 345 frames (14.375 s) is the longest clip there is.
    assert clip_plan.MAX_SECONDS == pytest.approx(14.375)
    assert 362 / clip_plan.FPS > 15.0


@pytest.mark.parametrize("frames", range(1, 400))
def test_align_frames_lands_on_the_chunk_grid(frames):
    aligned = clip_plan.align_frames(frames)
    assert aligned % 17 == 5
    assert aligned >= frames
    assert aligned - frames < 17


@pytest.mark.parametrize("seconds", [5.0, 5.167, 8.0, 10.0, 14.375, 15.0, 60.0])
def test_every_accepted_length_is_generatable(seconds):
    frames = clip_plan.frames_for_seconds(seconds)
    assert frames % 17 == 5
    assert clip_plan.MIN_FRAMES <= frames <= clip_plan.MAX_FRAMES
    assert 5.0 <= clip_plan.seconds_for_frames(frames) <= 15.0


def test_published_bounds_round_inward():
    """A client reading the schema sees tidy numbers that still snap cleanly."""
    assert clip_plan.MIN_SECONDS_PUBLISHED == 5.167
    assert clip_plan.MAX_SECONDS_PUBLISHED == 14.375
    assert clip_plan.MIN_SECONDS_PUBLISHED >= clip_plan.MIN_SECONDS
    assert clip_plan.MAX_SECONDS_PUBLISHED <= clip_plan.MAX_SECONDS
    for bound in (clip_plan.MIN_SECONDS_PUBLISHED, clip_plan.MAX_SECONDS_PUBLISHED):
        assert clip_plan.frames_for_seconds(bound) % 17 == 5


def test_seconds_must_be_positive():
    with pytest.raises(ValueError):
        clip_plan.frames_for_seconds(0)


@pytest.mark.parametrize("aspect", clip_plan.ASPECT_CHOICES)
def test_every_offered_canvas_satisfies_the_checkpoint(aspect):
    height, width = clip_plan.canvas_for_choice(aspect)
    assert height % 32 == 0 and width % 32 == 0
    assert height * width <= 768 * 1344
    assert min(height, width) == 768 or height * width <= 768 * 1344
    assert 1 / 4 <= width / height <= 4


def test_default_canvas_is_the_measured_one():
    assert clip_plan.canvas_for_choice("16:9") == (768, 1344)


def test_unknown_aspect_is_rejected():
    with pytest.raises(ValueError):
        clip_plan.canvas_for_choice("32:9")


def test_upstream_constants_have_not_drifted():
    """The duplicated constants must still match FastVideo's own.

    ``fasth3_clip_plan`` copies these rather than importing them, so that the
    schema renders without torch. This is the test that stops the copy going
    stale.
    """
    packing = pytest.importorskip(
        "fastvideo.pipelines.basic.minimax_h3.packing",
        reason="fastvideo is not installed on this machine",
    )
    assert clip_plan.FPS == packing.MINIMAX_H3_FPS
    assert clip_plan._FRAMES_PER_CHUNK == packing.MINIMAX_H3_FRAMES_PER_CHUNK
    assert clip_plan._LATENTS_PER_CHUNK == packing.MINIMAX_H3_LATENTS_PER_CHUNK
    assert clip_plan._MIN_DURATION == packing.MINIMAX_H3_MIN_DURATION
    assert clip_plan._MAX_DURATION == packing.MINIMAX_H3_MAX_DURATION
    assert clip_plan._SHORT_EDGE == packing.MINIMAX_H3_SHORT_EDGE
    assert clip_plan._MAX_PIXELS == packing.MINIMAX_H3_MAX_PIXELS
    assert clip_plan._CANVAS_MULTIPLE == packing.MINIMAX_H3_CANVAS_MULTIPLE

    for frames in (1, 100, 124, 200, 345):
        assert clip_plan.align_frames(frames) == packing.align_num_frames(frames)
    for aspect in clip_plan.ASPECT_CHOICES:
        ratio = clip_plan._ASPECT_RATIOS[aspect]
        assert clip_plan.canvas_for_choice(aspect) == packing.resolve_canvas_size(*ratio)


# ------------------------------------------------------------------ the queue
#
# Pure bookkeeping: order, capacity, and one wire form for every mention.


def make_queue(capacity=3) -> ClipQueue:
    return ClipQueue(capacity)


def add(q, prompt, seed=0, position=None, frames=124):
    entry = new_entry(prompt=prompt, metadata="", frames=frames, seed=seed)
    q.add(entry, position)
    return entry


def built(entry):
    entry.video = [np.zeros((2, 2, 3), np.uint8)]
    entry.audio = np.zeros((1, 4), np.int16)
    return entry


def test_the_queue_keeps_add_order():
    q = make_queue()
    a = add(q, "a", seed=1)
    b = add(q, "b", seed=2)
    assert [entry.prompt for entry in map(q.get, [a.clip_id, b.clip_id])] == ["a", "b"]
    assert q.snapshot()[0]["clip_id"] == a.clip_id
    assert q.next_to_build() is a
    assert q.head() is a


def test_add_at_a_position_lands_there_clamped():
    q = make_queue(capacity=5)
    add(q, "a")
    add(q, "b")
    add(q, "front", position=0)
    add(q, "middle", position=2)
    add(q, "past-the-end", position=99)
    assert [e["prompt"] for e in q.snapshot()] == [
        "front", "a", "middle", "b", "past-the-end",
    ]


def test_move_repositions_within_the_queue():
    q = make_queue(capacity=5)
    a, b, c = add(q, "a"), add(q, "b"), add(q, "c")
    assert q.move(c, 0) == 0
    assert [e["prompt"] for e in q.snapshot()] == ["c", "a", "b"]
    assert q.move(c, 99) == 2  # clamped to the back
    assert [e["prompt"] for e in q.snapshot()] == ["a", "b", "c"]
    q.remove(b)
    with pytest.raises(ValueError):
        q.move(b, 0)


def test_every_clip_gets_a_distinct_uuid():
    q = make_queue()
    ids = {add(q, "p").clip_id for _ in range(3)}
    assert len(ids) == 3


def test_the_queue_is_bounded():
    q = make_queue(capacity=2)
    add(q, "a", seed=1)
    add(q, "b", seed=2)
    assert q.full
    with pytest.raises(ValueError):
        add(q, "c", seed=3)


def test_ready_is_derived_from_the_built_payload():
    q = make_queue()
    entry = add(q, "a", seed=1)
    assert entry.ready is False
    built(entry)
    assert entry.ready is True


def test_building_entries_are_not_resubmitted():
    q = make_queue()
    entry = add(q, "a", seed=1)
    entry.building = True
    assert q.next_to_build() is None
    behind = add(q, "b", seed=2)
    assert q.next_to_build() is behind


def test_the_snapshot_is_exactly_the_published_struct():
    """`ClipEntry.snapshot()` and the schema's `ClipInfo` must never drift."""
    q = make_queue()
    entry = new_entry(prompt="a", metadata="m", frames=124, seed=7)
    q.add(entry)
    snapshot = entry.snapshot()
    assert list(snapshot) == [field.name for field in dataclasses.fields(ClipInfo)]
    assert snapshot["seconds"] == pytest.approx(124 / 24, abs=1e-3)
    assert snapshot["metadata"] == "m"
    assert snapshot["ready"] is False


# --------------------------------------------------------------- session rules
#
# The command state machine clients read out of `state_update`.


def rules(*, playing=False, generation=0, playout=0, capacity=10):
    return session_rules.valid_commands(
        playing=playing,
        generation_queued=generation,
        generation_capacity=capacity,
        playout_queued=playout,
    )


def test_an_empty_idle_session_can_only_enqueue_and_configure():
    commands = rules()
    assert "enqueue" in commands
    assert "set_canvas" in commands
    assert "play" not in commands
    assert "stop" not in commands
    assert "move" not in commands


def test_a_built_clip_makes_play_valid():
    commands = rules(playout=1)
    assert "play" in commands
    assert "pop" in commands
    assert "move" in commands
    # Queued clips were built at the current canvas, so it is locked.
    assert "set_canvas" not in commands


def test_pop_and_move_need_a_queued_clip():
    commands = rules()
    assert "pop" not in commands
    assert "move" not in commands
    assert "pop" in rules(generation=1)


def test_playing_offers_stop_and_locks_the_canvas():
    commands = rules(playing=True)
    assert "stop" in commands
    assert "play" not in commands
    assert "set_canvas" not in commands


def test_a_full_generation_queue_refuses_enqueue():
    commands = rules(generation=10, capacity=10)
    assert "enqueue" not in commands


def test_conditions_and_reads_are_always_available():
    for playing, generation, playout in ((False, 0, 0), (True, 3, 1), (False, 10, 10)):
        commands = rules(playing=playing, generation=generation, playout=playout)
        assert {
            "set_clip_seconds", "set_seed", "set_autoplay", "get_queue", "get_state", "reset"
        } <= set(commands)


# --------------------------------------------------------------------- config


def test_the_shipped_config_parses(tmp_path):
    config = load_config(MODEL_DIR / "fasth3.yaml")
    assert config.queue_size == 10
    assert config.aspect == "16:9"
    assert config.clip_frames == clip_plan.MAX_FRAMES
    # The shipped config warms every legal length: a feed of arbitrary
    # `seconds` values then never pays a first-use compile stall.
    assert config.warmup_frames == clip_plan.legal_frame_counts()


def test_every_legal_length_is_aligned_and_within_bounds():
    counts = clip_plan.legal_frame_counts()
    assert counts[0] == clip_plan.MIN_FRAMES
    assert counts[-1] == clip_plan.MAX_FRAMES
    assert all(frames % 17 == 5 for frames in counts)
    assert len(counts) == 14


def test_warmup_lengths_parse_all_shapes(tmp_path):
    default = tmp_path / "default.yaml"
    default.write_text("inference: {}\n", encoding="utf-8")
    config = load_config(default)
    assert config.warmup_frames == (config.clip_frames,)

    listed = tmp_path / "listed.yaml"
    listed.write_text(
        "inference:\n  warmup_lengths: [5.167, 8.0]\n", encoding="utf-8"
    )
    config = load_config(listed)
    # Snapped to legal lengths, default length always included, ascending.
    assert config.warmup_frames == tuple(sorted({
        clip_plan.frames_for_seconds(5.167),
        clip_plan.frames_for_seconds(8.0),
        config.clip_frames,
    }))

    bad = tmp_path / "bad.yaml"
    bad.write_text("inference:\n  warmup_lengths: sometimes\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)


def test_a_bad_aspect_or_queue_size_fails_startup(tmp_path):
    bad_aspect = tmp_path / "aspect.yaml"
    bad_aspect.write_text("inference:\n  aspect: '32:9'\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad_aspect)

    bad_queue = tmp_path / "queue.yaml"
    bad_queue.write_text("inference:\n  queue_size: 0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad_queue)


# ------------------------------------------------------------ command contract
#
# The real handlers on a model whose ``load()`` never ran: everything they touch
# is session state and pure arithmetic, so the whole state machine — refusals
# included — is testable on a laptop.


def make_config(queue_size=3, generation_queue_size=None) -> FastH3Config:
    return FastH3Config(
        aspect="16:9",
        clip_frames=clip_plan.frames_for_seconds(clip_plan.MAX_SECONDS),
        seed=1000,
        num_inference_steps=5,
        queue_size=queue_size,
        generation_queue_size=generation_queue_size or queue_size,
        warmup_aspects=("16:9",),
        warmup_frames=(clip_plan.frames_for_seconds(clip_plan.MAX_SECONDS),),
        inference={},
        runtime={},
    )


def run(coro):
    """Drive one handler to completion."""
    return asyncio.run(coro)


def refusal(model):
    """The most recent `command_error` broadcast, or None if there is none.

    A refusal is not an exception here: a handler reports failure by
    broadcasting `command_error` and returning without a value, so that clients
    on every SDK generation can see it.
    """
    errors = [message for message in model.sent if type(message).__name__ == "CommandError"]
    return errors[-1] if errors else None


@pytest.fixture
def model():
    """A FastH3 with the attributes ``load()`` would have set, and no engine."""
    instance = FastH3()
    # Loop-bound state the runtime creates when the model loop starts.
    instance._on_loop_ready()
    instance.config = make_config(queue_size=3)
    instance._reset_session_state()

    sent: list = []

    async def capture(message):
        sent.append(message)

    instance.send = capture
    instance.sent = sent
    return instance


def make_ready(model, clip):
    """What `_pump_builds` does when a build lands: cross into playout."""
    entry = model._generation.get(clip["clip_id"])
    model._generation.remove(entry)
    entry.video, entry.audio = [np.zeros((2, 2, 3), np.uint8)], np.zeros((1, 10), np.int16)
    model._playout.add(entry)
    return entry


def test_enqueue_returns_the_full_struct(model):
    reply = run(model.enqueue(prompt="a lighthouse in fog", metadata="req-42"))
    clip = reply.clip
    assert clip["prompt"] == "a lighthouse in fog"
    assert clip["metadata"] == "req-42"
    assert clip["ready"] is False
    assert clip["frames"] == model.config.clip_frames
    assert clip["seconds"] == pytest.approx(clip["frames"] / 24, abs=1e-3)
    assert clip["clip_id"]
    # The struct is a plain mapping, because the wire encoder takes only
    # JSON-representable values; ClipInfo is its schema-side declaration.
    assert list(clip) == [field.name for field in dataclasses.fields(ClipInfo)]


def test_enqueue_position_zero_is_the_next_build(model):
    first = run(model.enqueue(prompt="waits", metadata="")).clip
    jumped = run(model.enqueue(prompt="jumps", metadata="", position=0)).clip
    order = [clip["clip_id"] for clip in model._generation.snapshot()]
    assert order == [jumped["clip_id"], first["clip_id"]]
    assert model._generation.next_to_build().clip_id == jumped["clip_id"]


def test_move_repositions_and_names_the_queue(model):
    first = run(model.enqueue(prompt="a", metadata="")).clip
    second = run(model.enqueue(prompt="b", metadata="")).clip
    reply = run(model.move(clip_id=second["clip_id"], position=0))
    assert reply.queue == "generation" and reply.position == 0
    order = [clip["clip_id"] for clip in model._generation.snapshot()]
    assert order == [second["clip_id"], first["clip_id"]]

    built = make_ready(model, first)
    reply = run(model.move(clip_id=built.clip_id, position=0))
    assert reply.queue == "playout" and reply.position == 0

    assert run(model.move(clip_id="nope", position=0)) is None
    assert refusal(model).command == "move" 


def test_enqueue_needs_a_prompt(model):
    assert run(model.enqueue(prompt="   ", metadata="")) is None
    assert refusal(model).command == "enqueue"
    assert len(model._generation) == 0


def test_enqueue_snapshots_the_conditions_in_force(model):
    run(model.set_clip_seconds(seconds=8.0))
    first = run(model.enqueue(prompt="a", metadata="")).clip
    run(model.set_clip_seconds(seconds=14.375))
    second = run(model.enqueue(prompt="b", metadata="")).clip
    assert first["frames"] == clip_plan.frames_for_seconds(8.0)
    assert second["frames"] == clip_plan.frames_for_seconds(14.375)
    # Clips already queued keep the length they were enqueued with.
    assert model._generation.get(first["clip_id"]).frames == first["frames"]


def test_each_enqueue_advances_the_seed(model):
    seeds = [run(model.enqueue(prompt="p", metadata="")).clip["seed"] for _ in range(2)]
    assert seeds == [1000, 1001]
    run(model.reset())
    run(model.set_seed(seed=7))
    assert run(model.enqueue(prompt="p", metadata="")).clip["seed"] == 7


def test_an_explicit_length_applies_to_that_clip_only(model):
    """A per-enqueue `seconds` snaps to the grid and spares the session default."""
    clip = run(model.enqueue(prompt="p", metadata="", seconds=8.3)).clip
    assert clip["frames"] == clip_plan.frames_for_seconds(8.3)
    assert clip["frames"] % 17 == 5
    assert model._clip_frames == model.config.clip_frames  # default untouched
    plain = run(model.enqueue(prompt="p", metadata="")).clip
    assert plain["frames"] == model.config.clip_frames


def test_an_explicit_seed_leaves_the_default_untouched(model):
    """Explicit and automatic seeding must not interfere with each other."""
    explicit = run(model.enqueue(prompt="p", metadata="", seed=42)).clip
    assert explicit["seed"] == 42
    assert model._seed == 1000  # the advancing default did not move
    automatic = run(model.enqueue(prompt="p", metadata="")).clip
    assert automatic["seed"] == 1000
    assert model._seed == 1001


def test_autoplay_is_a_session_condition(model):
    assert run(model.get_state()).autoplay is False
    reply = run(model.set_autoplay(enabled=True))
    assert reply.enabled is True
    assert run(model.get_state()).autoplay is True
    # `reset` returns every condition to its default, autoplay included.
    run(model.reset())
    assert run(model.get_state()).autoplay is False


def test_a_full_queue_refuses_the_next_enqueue(model):
    for index in range(3):
        run(model.enqueue(prompt=f"p{index}", metadata=""))
    assert run(model.enqueue(prompt="overflow", metadata="")) is None
    assert refusal(model).command == "enqueue"
    assert len(model._generation) == 3


def test_play_needs_a_ready_clip(model):
    assert run(model.play(clip_id="")) is None
    assert refusal(model).command == "play"

    queued = run(model.enqueue(prompt="p", metadata="")).clip
    assert run(model.play(clip_id=queued["clip_id"])) is None  # enqueued, not built
    assert refusal(model).reason.startswith("That clip is still generating")

    assert run(model.play(clip_id="not-a-real-id")) is None
    assert "not-a-real-id" in refusal(model).reason


def test_play_takes_the_playout_front(model):
    first = run(model.enqueue(prompt="a", metadata="")).clip
    second = run(model.enqueue(prompt="b", metadata="")).clip
    make_ready(model, first)
    make_ready(model, second)

    run(model.play(clip_id=""))
    assert model._play_request.clip_id == first["clip_id"]
    # Playing consumed the entry: the playout queue holds only the second.
    assert [clip["clip_id"] for clip in model._playout.snapshot()] == [second["clip_id"]]


def test_play_by_id_takes_that_clip(model):
    run(model.enqueue(prompt="a", metadata=""))
    second = run(model.enqueue(prompt="b", metadata="")).clip
    make_ready(model, second)

    run(model.play(clip_id=second["clip_id"]))
    assert model._play_request.clip_id == second["clip_id"]


def test_only_one_clip_plays_at_a_time(model):
    queued = run(model.enqueue(prompt="a", metadata="")).clip
    make_ready(model, queued)
    run(model.play(clip_id=""))

    assert run(model.play(clip_id="")) is None
    assert refusal(model).reason.startswith("A clip is already playing")


def test_pop_frees_the_slot(model):
    kept = run(model.enqueue(prompt="keep", metadata="")).clip
    victim = run(model.enqueue(prompt="drop", metadata="")).clip
    reply = run(model.pop(clip_id=victim["clip_id"]))
    assert reply.clip["clip_id"] == victim["clip_id"]
    assert [c["clip_id"] for c in model._generation.snapshot()] == [kept["clip_id"]]

    # And a built clip pops out of the playout queue the same way.
    make_ready(model, kept)
    reply = run(model.pop(clip_id=kept["clip_id"]))
    assert reply.clip["clip_id"] == kept["clip_id"]
    assert len(model._playout) == 0

    assert run(model.pop(clip_id="nope")) is None
    assert refusal(model).command == "pop"
    assert run(model.pop(clip_id="")) is None  # the id is required
    assert refusal(model).command == "pop"


def test_pop_cancels_the_build_in_flight(model):
    from fasth3_backend import ClipJob

    clip = run(model.enqueue(prompt="building", metadata="")).clip
    entry = model._generation.get(clip["clip_id"])
    entry.building = True
    job = ClipJob(None)
    model._build = (entry, job, 0.0)

    run(model.pop(clip_id=clip["clip_id"]))
    assert job.cancelled is True
    assert len(model._generation) == 0


def test_stop_needs_a_playing_clip(model):
    assert run(model.stop()) is None
    assert refusal(model).command == "stop"
    assert model._stop_playout is False


def test_stop_asks_the_playout_loop_to_cut(model):
    queued = run(model.enqueue(prompt="a", metadata="")).clip
    make_ready(model, queued)
    run(model.play(clip_id=""))

    run(model.stop())
    assert model._stop_playout is True
    # The queues are untouched: stop cuts playout, not the queues.
    assert len(model._playout) == 0  # the played clip had already left it


def test_the_canvas_is_locked_while_clips_exist(model):
    reply = run(model.set_canvas(aspect="9:16"))
    assert (reply.height, reply.width) == clip_plan.canvas_for_choice("9:16")

    run(model.enqueue(prompt="a", metadata=""))
    assert run(model.set_canvas(aspect="1:1")) is None
    assert refusal(model).command == "set_canvas"
    # The refused command had no effect.
    assert model._aspect == "9:16"

    run(model.reset())
    assert run(model.set_canvas(aspect="1:1")) is not None


def test_clip_length_snaps_to_something_generatable(model):
    reply = run(model.set_clip_seconds(seconds=8.3))
    assert reply.frames % 17 == 5
    assert reply.clip_seconds == pytest.approx(reply.frames / 24, abs=1e-3)
    assert model._clip_frames == reply.frames


def test_reset_drops_the_queue_and_restores_every_default(model):
    run(model.set_clip_seconds(seconds=10.0))
    run(model.set_seed(seed=7))
    run(model.enqueue(prompt="a", metadata=""))
    run(model.enqueue(prompt="b", metadata=""))

    reply = run(model.reset())
    assert reply.cleared_clips == 2
    assert reply.was_playing is False
    assert len(model._generation) == 0
    assert len(model._playout) == 0
    assert model._clip_frames == model.config.clip_frames
    assert model._seed == model.config.seed
    assert model._aspect == model.config.aspect


def test_get_queue_reports_the_same_payload_that_is_broadcast(model):
    run(model.enqueue(prompt="a", metadata="m"))
    direct = run(model.get_queue())
    broadcasts = [m for m in model.sent if type(m).__name__ == "QueueUpdate"]
    assert direct.generation == broadcasts[-1].generation
    assert direct.playout == broadcasts[-1].playout
    assert direct.generation[0]["metadata"] == "m"


def test_get_state_reports_the_same_snapshot_that_is_broadcast(model):
    run(model.enqueue(prompt="a", metadata=""))
    model.sent.clear()
    run(model._send_state_update())
    broadcast = model.sent[-1]
    direct = run(model.get_state())
    assert vars(direct) == vars(broadcast)


def test_the_snapshot_publishes_the_live_command_set(model):
    snapshot = run(model.get_state())
    assert "play" not in snapshot.valid_commands  # nothing built yet
    assert "enqueue" in snapshot.valid_commands
    assert snapshot.generation_queued == 0
    assert snapshot.generation_capacity == 3
    assert snapshot.playout_queued == 0
    assert snapshot.playout_capacity == 3
    assert snapshot.playing is False
    assert snapshot.playing_clip_id is None

    queued = run(model.enqueue(prompt="a", metadata="")).clip
    make_ready(model, queued)
    snapshot = run(model.get_state())
    assert "play" in snapshot.valid_commands
    assert snapshot.generation_queued == 0
    assert snapshot.playout_queued == 1


def test_a_refusal_never_masquerades_as_a_reply(model):
    """A handler must return only the type its annotation names.

    `enqueue` is annotated `-> ClipQueued`; a refusal that returned a
    `CommandError` from it would reach the client typed as the message the
    schema promised, with every guaranteed field undefined.
    """
    assert run(model.enqueue(prompt="", metadata="")) is None
    error = refusal(model)
    assert type(error).__name__ == "CommandError"
    assert error.reason


def test_the_runtime_exception_is_never_raised(model):
    """Its correlated failure frame is withheld from v0 clients, so a raise is
    silence for anyone on the 2.x SDK. The broadcast is what reaches them."""
    source = (MODEL_DIR / "fasth3.py").read_text(encoding="utf-8")
    assert "raise CommandError" not in source
    # And the name in scope is the model's own message, not the runtime's.
    from fasth3 import CommandError as in_scope
    from fasth3_types import CommandError as ours

    assert in_scope is ours


# ------------------------------------------------------------ the playout loop
#
# ``_serve`` is the concurrent part of this model — a worker building clips, an
# event loop pacing an armed clip out, and abort paths crossing both. These
# tests replace only the *backend*, so the real queueing, ordering, emission
# and teardown all run.
#
# Clips are shrunk to a few frames so a whole playout runs in well under a
# second; the pacer is a real 24 fps clock, so the numbers below are wall time.

FRAMES_PER_CLIP = 6  # 0.25 s of content at 24 fps


class FakeBackend:
    """Builds tiny clips on demand; instant by default, controllable when not."""

    def __init__(self):
        self.built: list[tuple[int, str, int]] = []
        self.fail_next: Exception | None = None
        self.hold = False
        self.held: list[ClipJob] = []

    def submit(self, *, frames, prompt, seed, height, width) -> ClipJob:
        self.built.append((frames, prompt, seed))
        job = ClipJob(None)
        if self.fail_next is not None:
            job.error, self.fail_next = self.fail_next, None
            job.done.set()
        elif self.hold:
            self.held.append(job)
        else:
            self.finish(job, frames)
        return job

    @staticmethod
    def finish(job: ClipJob, frames: int = FRAMES_PER_CLIP) -> None:
        video = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(frames)]
        samples = np.zeros((1, round(frames / 24 * 48_000)), dtype=np.int16)
        job.result = (video, samples)
        job.done.set()


@pytest.fixture
def live():
    """A FastH3 wired to a fake backend, with a connected audience."""
    instance = FastH3()
    instance._on_loop_ready()
    instance.connected.set()
    instance.config = make_config(queue_size=5)
    instance._reset_session_state()
    instance._clip_frames = FRAMES_PER_CLIP  # tiny clips keep the tests fast
    instance.backend = FakeBackend()

    emitted: list = []

    async def fake_emit(output):
        emitted.append(output)

    instance.emit = fake_emit
    instance.emitted = emitted

    messages: list = []

    async def fake_send(message):
        messages.append(message)

    instance.send = fake_send
    instance.messages = messages

    flushes: list = []
    instance.output.flush = lambda: flushes.append(time.monotonic())
    instance.flushes = flushes
    return instance


def names(messages) -> list[str]:
    return [type(m).__name__ for m in messages]


def drive(live, scenario):
    """Run `_serve` against a scenario coroutine that ends the session."""

    async def main():
        async def wrapped():
            try:
                await scenario()
            finally:
                live.connected.clear()

        await asyncio.gather(live._serve(), wrapped())

    asyncio.run(main())


async def eventually(predicate, timeout=2.0):
    """Wait until `predicate()` is true, failing the test after `timeout`."""
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition never became true"
        await asyncio.sleep(0.005)


def test_enqueued_clips_build_in_order_and_turn_ready(live):
    async def scenario():
        await live.enqueue(prompt="a", metadata="")
        await live.enqueue(prompt="b", metadata="")
        await eventually(lambda: len(live._playout) == 2)

    drive(live, scenario)
    assert [prompt for _f, prompt, _s in live.backend.built] == ["a", "b"]
    ready = [m for m in live.messages if type(m).__name__ == "QueueUpdate"]
    assert ready, "no queue_update announced the clips turning ready"


def test_a_played_clip_streams_whole_then_holds_on_black(live):
    async def scenario():
        await live.enqueue(prompt="a", metadata="tag")
        await eventually(lambda: len(live._playout) == 1)
        await live.play(clip_id="")
        await eventually(lambda: "ClipFinished" in names(live.messages))
        # No auto-play: nothing else may start on its own.
        await asyncio.sleep(0.15)

    drive(live, scenario)
    # `clip_queued` is the enqueue reply, not a broadcast, so it is not here;
    # `clip_generated` marks the build crossing into the playout queue.
    assert names([m for m in live.messages if "Clip" in type(m).__name__]) == [
        "ClipGenerated",
        "ClipStarted",
        "ClipFinished",
    ]
    frames = sum(output.main_video.shape[0] for output in live.emitted)
    assert frames == FRAMES_PER_CLIP
    assert live.flushes, "the stream must flush to black after the clip"
    assert live._playing is None
    # The full struct rides on both playout messages, metadata included.
    started = next(m for m in live.messages if type(m).__name__ == "ClipStarted")
    finished = next(m for m in live.messages if type(m).__name__ == "ClipFinished")
    assert started.clip["metadata"] == "tag"
    assert started.clip["ready"] is True
    assert finished.clip["clip_id"] == started.clip["clip_id"]


def test_video_and_audio_stay_locked_slice_for_slice(live):
    async def scenario():
        await live.enqueue(prompt="a", metadata="")
        await eventually(lambda: len(live._playout) == 1)
        await live.play(clip_id="")
        await eventually(lambda: "ClipFinished" in names(live.messages))

    drive(live, scenario)
    for output in live.emitted:
        video_frames = output.main_video.shape[0]
        audio_samples = output.main_audio.shape[1]
        assert output.main_video.dtype == np.uint8
        assert output.main_audio.dtype == np.int16
        assert output.main_audio.ndim == 2 and output.main_audio.shape[0] == 1
        assert audio_samples == pytest.approx(video_frames * 48_000 / 24, abs=1)


def test_stop_cuts_the_clip_and_keeps_the_queue(live):
    # A two-second clip, so the stop reliably lands mid-play.
    live._clip_frames = 48

    async def scenario():
        await live.enqueue(prompt="a", metadata="")
        await live.enqueue(prompt="b", metadata="")
        await eventually(lambda: len(live._playout) == 2)
        await live.play(clip_id="")
        await eventually(lambda: live.emitted)
        await live.stop()
        await eventually(lambda: "ClipStopped" in names(live.messages))

    drive(live, scenario)
    assert "ClipFinished" not in names(live.messages)
    assert live.flushes, "stop must flush the output"
    # The cut clip went out only partially.
    frames = sum(output.main_video.shape[0] for output in live.emitted)
    assert frames < 48
    # The other clip still waits, built, for the next play.
    assert len(live._playout) == 1


def test_builds_continue_while_a_clip_plays(live):
    # A two-second clip, so the enqueue reliably lands mid-play.
    live._clip_frames = 48
    built_during_play = {}

    async def scenario():
        await live.enqueue(prompt="a", metadata="")
        await eventually(lambda: len(live._playout) == 1)
        await live.play(clip_id="")
        await eventually(lambda: live.emitted)
        await live.enqueue(prompt="b", metadata="")
        await eventually(lambda: len(live._playout) == 1)
        built_during_play["value"] = live._playing is not None
        await eventually(lambda: "ClipFinished" in names(live.messages))

    drive(live, scenario)
    # The second build was submitted and finished while the first clip streamed.
    assert [prompt for _f, prompt, _s in live.backend.built] == ["a", "b"]
    assert built_during_play["value"] is True
    assert len(live._playout) == 1  # the mid-play build crossed into playout


def test_a_failing_build_reports_and_the_queue_moves_on(live):
    async def scenario():
        live.backend.fail_next = RuntimeError("the engine fell over")
        await live.enqueue(prompt="a", metadata="")
        await live.enqueue(prompt="b", metadata="")
        await eventually(lambda: "ClipFailed" in names(live.messages))
        await eventually(lambda: len(live._playout) == 1)

    drive(live, scenario)
    failed = next(m for m in live.messages if type(m).__name__ == "ClipFailed")
    assert "the engine fell over" in failed.reason
    assert failed.clip["prompt"] == "a"
    # The failed clip left the queues; the survivor crossed into playout.
    assert [clip["prompt"] for clip in live._playout.snapshot()] == ["b"]


def test_reset_discards_a_build_still_in_flight(live):
    async def scenario():
        live.backend.hold = True
        await live.enqueue(prompt="a", metadata="")
        await eventually(lambda: live.backend.held)
        await live.reset()
        live.backend.finish(live.backend.held[0])
        # The finished build has no entry to land on; nothing may surface.
        await asyncio.sleep(0.15)

    drive(live, scenario)
    assert len(live._generation) == 0
    assert len(live._playout) == 0
    assert "ClipFailed" not in names(live.messages)


def test_the_pacer_holds_24_fps(live):
    elapsed = {}

    async def scenario():
        await live.enqueue(prompt="a", metadata="")
        await eventually(lambda: len(live._playout) == 1)
        started = time.monotonic()
        await live.play(clip_id="")
        await eventually(lambda: "ClipFinished" in names(live.messages))
        elapsed["playout"] = time.monotonic() - started

    drive(live, scenario)
    content = FRAMES_PER_CLIP / 24
    # A slice is handed over at the instant its content is due, so the playout
    # ends one slice before the content it queued finishes playing.
    expected = content - EMIT_FRAMES / 24
    # Real-time, not faster: the metronome is what keeps the audio in sync.
    assert elapsed["playout"] >= expected * 0.95
    assert elapsed["playout"] < content + 0.3


def test_autoplay_chains_ready_clips_without_play(live):
    async def scenario():
        await live.set_autoplay(enabled=True)
        await live.enqueue(prompt="a", metadata="")
        await live.enqueue(prompt="b", metadata="")
        # No `play` anywhere in this scenario: both clips must stream on
        # their own, oldest first, once their builds complete.
        await eventually(
            lambda: names(live.messages).count("ClipFinished") == 2, timeout=5.0
        )
        # The queue is drained and nothing else may start.
        await asyncio.sleep(0.15)

    drive(live, scenario)
    started = [m.clip["prompt"] for m in live.messages if type(m).__name__ == "ClipStarted"]
    assert started == ["a", "b"]
    frames = sum(output.main_video.shape[0] for output in live.emitted)
    assert frames == 2 * FRAMES_PER_CLIP
    assert live.flushes, "the stream still flushes to black at each boundary"


def test_without_autoplay_nothing_starts_on_its_own(live):
    async def scenario():
        await live.enqueue(prompt="a", metadata="")
        await eventually(lambda: len(live._playout) == 1)
        await asyncio.sleep(0.2)

    drive(live, scenario)
    assert "ClipStarted" not in names(live.messages)
    assert live.emitted == []


def test_a_lost_audience_ends_the_playout_quietly(live):
    # A two-second clip, so the disconnect reliably lands mid-play.
    live._clip_frames = 48

    async def scenario():
        await live.enqueue(prompt="a", metadata="")
        await eventually(lambda: len(live._playout) == 1)
        await live.play(clip_id="")
        await eventually(lambda: live.emitted)
        # The scenario wrapper clears `connected`, which is the audience leaving.

    drive(live, scenario)
    # Nobody was there to hear a finish or a stop for the cut clip.
    assert "ClipFinished" not in names(live.messages)
    assert "ClipStopped" not in names(live.messages)


def test_generation_is_gated_on_an_audience(live):
    """`_serve` is the only submitter, and it runs only with a client connected."""

    async def main():
        await live.enqueue(prompt="a", metadata="")
        live.connected.clear()
        await asyncio.wait_for(live._serve(), timeout=1.0)

    asyncio.run(main())
    assert live.backend.built == []
    assert len(live._playout) == 0


# ---------------------------------------------------------- published contract
#
# ``reactor schema`` compiles this document out of the model class, and
# generated SDKs are built from nothing else. A change here is a change to every
# client, so these tests exist to make an accidental one fail loudly and a
# deliberate one obvious in review — together with the version bump it needs.

# Every command a client can send, and the message each answers with. `None`
# means the command is answered with a bare acknowledgement.
EXPECTED_COMMANDS = {
    "enqueue": "ClipQueued",
    "get_queue": "QueueUpdate",
    "get_state": "StateUpdate",
    "move": "ClipMoved",
    "play": None,
    "pop": "ClipPopped",
    "reset": "SessionReset",
    "set_autoplay": "AutoplayAccepted",
    "set_canvas": "CanvasAccepted",
    "set_clip_seconds": "ClipLengthAccepted",
    "set_seed": "SeedAccepted",
    "stop": None,
}

EXPECTED_MESSAGES = {
    "autoplay_accepted",
    "canvas_accepted",
    "clip_failed",
    "clip_finished",
    "clip_generated",
    "clip_length_accepted",
    "clip_moved",
    "clip_popped",
    "clip_queued",
    "clip_started",
    "clip_stopped",
    "command_error",
    "queue_update",
    "seed_accepted",
    "session_reset",
    "state_update",
}

# Commands that can be refused. Each one has to say so in its own summary, and
# name the message a client will actually receive.
EXPECTED_REJECTIONS = ("enqueue", "move", "play", "pop", "stop", "set_canvas")

# The struct every clip-referencing message embeds, and its JSON types.
EXPECTED_CLIP_INFO = {
    "clip_id": "string",
    "prompt": "string",
    "metadata": "string",
    "frames": "integer",
    "seconds": "number",
    "seed": "integer",
    "ready": "boolean",
}


@pytest.fixture(scope="module")
def schema():
    """Render the document exactly as the release pipeline does.

    The renderer imports the model without loading it, so this needs no weights
    and no GPU — but it does need the model's own imports to stay light, which
    is itself part of what this asserts.
    """
    from reactor_runtime.schema import render

    return render(MODEL_DIR, version="v0.2.0")


def test_the_model_publishes_two_outbound_tracks(schema):
    tracks = schema["x-reactor"]["tracks"]
    assert [(t["name"], t["kind"], t["direction"]) for t in tracks] == [
        ("main_video", "video", "out"),
        ("main_audio", "audio", "out"),
    ]


def test_the_command_set_is_exactly_what_clients_expect(schema):
    published = {path.removeprefix("/events/") for path in schema["paths"]}
    assert published == set(EXPECTED_COMMANDS)


def test_every_command_answers_with_the_type_it_promises(schema):
    for name, message in EXPECTED_COMMANDS.items():
        operation = schema["paths"][f"/events/{name}"]["post"]
        responses = operation["responses"]
        if message is None:
            # A handler that returns nothing is answered with a bare 202, so an
            # awaiting client still resolves — it just learns nothing.
            assert set(responses) == {"202"}, name
            continue
        body = responses["200"]["content"]["application/json"]["schema"]
        assert body["$ref"] == f"#/components/schemas/{message}", name


def test_every_message_is_published_once(schema):
    assert set(schema["webhooks"]) == EXPECTED_MESSAGES
    for message in EXPECTED_COMMANDS.values():
        if message is not None:
            assert message in schema["components"]["schemas"]


def test_every_clip_message_embeds_the_full_struct(schema):
    """`ClipInfo` rides whole on every message that references a clip."""
    for message in (
        "ClipQueued", "ClipGenerated", "ClipMoved", "ClipStarted",
        "ClipFinished", "ClipStopped", "ClipFailed", "ClipPopped",
    ):
        clip = schema["components"]["schemas"][message]["properties"]["clip"]
        rendered = {
            name: field["type"] for name, field in clip["properties"].items()
        }
        assert rendered == EXPECTED_CLIP_INFO, message
        assert set(clip.get("required", [])) == set(EXPECTED_CLIP_INFO), message
    # And both queues report lists of the same struct.
    for queue in ("generation", "playout"):
        items = schema["components"]["schemas"]["QueueUpdate"]["properties"][queue]["items"]
        assert {name: field["type"] for name, field in items["properties"].items()} == (
            EXPECTED_CLIP_INFO
        ), queue


def test_free_text_fields_are_marked_for_moderation(schema):
    """`enqueue` carries client free text into generated video and audio."""
    properties = schema["paths"]["/events/enqueue"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]
    assert properties["prompt"]["x-reactor-moderate"] is True
    assert properties["metadata"]["x-reactor-moderate"] is True


def test_the_enqueue_seed_and_length_are_optional_on_the_wire(schema):
    """Omitted or null means the session's defaults."""
    properties = schema["paths"]["/events/enqueue"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]
    for name in ("seed", "seconds"):
        field = properties[name]
        types = field.get("anyOf") or [field]
        assert any(entry.get("type") == "null" for entry in types), (name, field)


def test_the_clip_length_bounds_a_client_reads_are_generatable(schema):
    seconds = schema["paths"]["/events/set_clip_seconds"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]["seconds"]
    assert seconds["minimum"] == clip_plan.MIN_SECONDS_PUBLISHED
    assert seconds["maximum"] == clip_plan.MAX_SECONDS_PUBLISHED
    for bound in (seconds["minimum"], seconds["maximum"]):
        assert clip_plan.frames_for_seconds(bound) % 17 == 5


def test_the_canvas_choices_are_published_as_an_enum(schema):
    aspect = schema["paths"]["/events/set_canvas"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]["aspect"]
    assert aspect["enum"] == list(clip_plan.ASPECT_CHOICES)


def test_every_client_facing_string_is_documented(schema):
    """A frontend developer who cannot read this repo works from these alone."""
    for path, operations in schema["paths"].items():
        assert operations["post"].get("summary"), f"{path} has no description"
        body = operations["post"].get("requestBody")
        if not body:
            continue
        properties = body["content"]["application/json"]["schema"]["properties"]
        for name, field in properties.items():
            assert field.get("description"), f"{path} parameter {name} has no description"
    for name, message in schema["webhooks"].items():
        assert message["post"].get("summary"), f"message {name} has no description"
    # Only the messages this model declares. The runtime contributes its own
    # components (the upload reference, for one), and those are not ours to
    # document.
    ours = {
        message["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"].rsplit(
            "/", 1
        )[-1]
        for message in schema["webhooks"].values()
    }
    assert ours, "no message definitions were resolved from the webhooks"
    for name in ours:
        definition = schema["components"]["schemas"][name]
        for field_name, field in definition.get("properties", {}).items():
            assert field.get("description"), f"{name}.{field_name} has no description"


def test_every_message_summary_says_when_it_is_emitted(schema):
    """The house style: a message docstring opens with "Emitted ..."."""
    for name, message in schema["webhooks"].items():
        summary = message["post"]["summary"]
        assert summary.startswith("Emitted "), f"{name}: {summary!r}"


def test_no_message_name_repeats_the_model_name(schema):
    """The SDK client already identifies the model; a prefix is dead weight."""
    for name in schema["webhooks"]:
        assert not name.startswith("fasth3"), name
    for name in schema["components"]["schemas"]:
        assert not name.lower().startswith("fasth3"), name


def test_every_command_summary_names_what_it_emits(schema):
    for name in EXPECTED_COMMANDS:
        summary = schema["paths"][f"/events/{name}"]["post"]["summary"]
        if name == "get_state":
            # A pure read: it answers, it does not emit.
            assert "state_update" in summary
            continue
        if name == "get_queue":
            assert "queue_update" in summary
            continue
        if name == "move":
            # A move changes order, not state: the snapshot carries counts
            # and conditions, none of which a reposition touches.
            assert "queue_update" in summary
            continue
        assert "`state_update`" in summary, f"{name} does not say it broadcasts a snapshot"


def test_every_refusable_command_documents_its_failure(schema):
    for name in EXPECTED_REJECTIONS:
        summary = schema["paths"][f"/events/{name}"]["post"]["summary"]
        assert "`command_error`" in summary, f"{name} does not document its failure"


def test_the_idle_stream_id_is_null_not_empty(schema):
    """`None` is the no-clip value on the wire; an empty string would be ambiguous."""
    playing = schema["components"]["schemas"]["StateUpdate"]["properties"]["playing_clip_id"]
    types = playing.get("anyOf") or [playing]
    assert any(entry.get("type") == "null" for entry in types), playing


def test_clip_length_prose_matches_the_published_bounds(schema):
    """The numbers in the description are generated from the same constants."""
    summary = schema["paths"]["/events/set_clip_seconds"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]["seconds"]["description"]
    assert f"{clip_plan.MIN_SECONDS_PUBLISHED:g}" in summary
    assert f"{clip_plan.MAX_SECONDS_PUBLISHED:g}" in summary


# ------------------------------------------------------------------- manifest
#
# The workspace rules from GUIDELINES.md, checked here rather than discovered
# during a build.

# A stray checkpoint in the folder is a large, slow mistake to discover later.
WEIGHT_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx"}


@pytest.fixture(scope="module")
def manifest() -> dict:
    return yaml.safe_load((MODEL_DIR / "reactor.yaml").read_text(encoding="utf-8"))


def test_the_model_name_matches_the_folder(manifest):
    """The folder is the workspace, and the name is the published slug."""
    assert manifest["model"]["name"] == MODEL_DIR.name


def test_the_version_is_semver_with_a_v_prefix(manifest):
    version = manifest["model"]["version"]
    assert isinstance(version, str), "quote the version so it is not parsed as a number"
    assert re.fullmatch(r"v\d+\.\d+\.\d+", version), f"{version!r} must be semver with a `v`"


def test_the_manifest_carries_a_complete_resource_spec(manifest):
    resources = manifest["model"]["resources"]
    assert resources["gpu"]["type"] and resources["gpu"]["count"] >= 1
    assert resources["cpu"]["request"] and resources["cpu"]["limit"]
    assert resources["memory"]["request"] and resources["memory"]["limit"]


def test_the_image_is_built_from_the_manifest_not_a_dockerfile(manifest):
    """`reactor build` owns the image; a Dockerfile here would be ignored."""
    assert not (MODEL_DIR / "Dockerfile").exists()
    build = manifest["build"]
    assert build["python_requirements"] == "requirements.txt"
    assert (MODEL_DIR / build["python_requirements"]).is_file()


def test_the_config_the_runtime_hands_to_load_exists(manifest):
    config = manifest["runtime"]["config"]
    assert (MODEL_DIR / config).is_file(), f"runtime.config points at a missing {config}"


def test_the_runtime_pin_is_current(manifest):
    assert manifest["build"]["runtime_version"] == "3.2.6"


def test_the_runtime_release_is_pinned_once(manifest):
    """`build.runtime_version` owns it; a second pin would let the two drift."""
    assert "reactor-runtime" not in (MODEL_DIR / "requirements.txt").read_text(encoding="utf-8")


def test_the_runtime_import_resolves_to_the_model_class(manifest):
    module_name, _, class_name = manifest["runtime"]["import"].partition(":")
    module = __import__(module_name)
    assert getattr(module, class_name, None) is not None, f"{module_name}.py has no {class_name}"


def test_no_weights_are_committed_alongside_the_model():
    offenders = [
        path.relative_to(MODEL_DIR)
        for path in MODEL_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in WEIGHT_SUFFIXES
        # Hidden directories (.venv, .git) are not part of the commit.
        and not any(part.startswith(".") for part in path.relative_to(MODEL_DIR).parts)
    ]
    assert not offenders, f"weights never live in git: {offenders}"
