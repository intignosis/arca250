"""The Reactor side: one supervised connection to a served fasth3 model.

`ReactorLink` owns everything that touches `reactor_sdk`:

  * the connect/reconnect loop — a dropped session is rebuilt from scratch
    (queue contents die with a session server-side; see the README), while
    the pacer and sink outside this class keep the broadcast alive;
  * the media path — the recvonly video and audio tracks feed the pacer;
  * a live mirror of the model's `state_update` / `queue_update`, so the
    rest of the client reads state instead of re-deriving it;
  * a fan-out of every model message to registered listeners.

The model contract this speaks is the fasth3 clip queue (`../fasth3/fasth3_types.py`
is the authoritative reference): `enqueue` → `clip_queued`, readiness on
`queue_update`, autoplay for hands-free sequential playback, black between
clips. The link turns autoplay on right after every (re)connect, which is
what makes playback fully queue-driven: enqueue order is play order.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from reactor_sdk import Reactor, ReactorStatus

from config import Config
from pacer import Pacer

logger = logging.getLogger(__name__)

RECONNECT_DELAY_S = 5.0

# fasth3's fixed output timing (fasth3_clip_plan.FPS / backend sample rate).
# The canvas (width/height) is read from state_update instead — it depends on
# the deployment's aspect — but the rates are pinned by the checkpoint.
MODEL_FPS = 24
MODEL_SAMPLE_RATE = 48_000

# Defaults used only until the first state_update arrives.
_DEFAULT_STATE: dict[str, Any] = {
    "width": 1344,
    "height": 768,
    "clip_seconds_min": 5.167,
    "clip_seconds_max": 14.375,
    "queued": 0,
    "queue_capacity": 10,
}


def payload(reply: Any) -> Any:
    """Unwrap a send_command reply envelope ({"type", "data"}) to its data."""
    if isinstance(reply, dict) and "data" in reply and "type" in reply:
        return reply["data"]
    return reply


class ReactorLink:
    """Supervised fasth3 session: media into the pacer, commands out."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._pacer: Pacer | None = None
        self._reactor: Reactor | None = None
        self._ready = asyncio.Event()
        self._first_state = asyncio.Event()
        self._listeners: list[Callable[[str, dict], None]] = []
        self.state: dict[str, Any] = dict(_DEFAULT_STATE)
        self.queue_clips: list[dict] = []

    # -------------------------------------------------------------- wiring

    def attach_pacer(self, pacer: Pacer) -> None:
        """Point the media path at the pacer (built after the first state)."""
        self._pacer = pacer

    def add_listener(self, listener: Callable[[str, dict], None]) -> None:
        """Register for every model message as `(kind, data)`. Must not raise."""
        self._listeners.append(listener)

    # ------------------------------------------------------- state mirror

    @property
    def min_seconds(self) -> float:
        return float(self.state.get("clip_seconds_min", 5.167))

    @property
    def max_seconds(self) -> float:
        return float(self.state.get("clip_seconds_max", 14.375))

    @property
    def queued(self) -> int:
        return int(self.state.get("queued", 0))

    @property
    def queue_capacity(self) -> int:
        return int(self.state.get("queue_capacity", 10))

    @property
    def canvas(self) -> tuple[int, int]:
        """(width, height) the deployment generates at."""
        return int(self.state["width"]), int(self.state["height"])

    @property
    def connected(self) -> bool:
        """Whether a session is live right now (commands would go through)."""
        return self._ready.is_set()

    async def wait_first_state(self) -> None:
        """Resolve once the first session delivered its `state_update`."""
        await self._first_state.wait()

    def _on_message(self, message: Any) -> None:
        if not isinstance(message, dict):
            return
        kind = message.get("type")
        data = message.get("data") or {}
        if kind == "state_update":
            self.state = data
        elif kind == "queue_update":
            self.queue_clips = data.get("clips", [])
        elif kind == "command_error":
            logger.warning(
                "[reactor] command refused: %s — %s",
                data.get("command"), data.get("reason"),
            )
        for listener in self._listeners:
            listener(kind, data)

    # ------------------------------------------------------------ commands

    async def send_command(self, command: str, data: dict) -> Any:
        """Send one command on the live session; None when disconnected.

        Waits for a session to exist first, so callers ride out a reconnect
        instead of failing. A None / bodyless reply means the model refused
        the command (it broadcast `command_error` with the reason).
        """
        await self._ready.wait()
        reactor = self._reactor
        if reactor is None:
            return None
        try:
            return payload(await reactor.send_command(command, data))
        except Exception as error:
            logger.warning("[reactor] %s failed: %s", command, error)
            return None

    # ----------------------------------------------------------- lifecycle

    async def run(self) -> None:
        """Connect, and keep reconnecting forever. Cancelled only at shutdown."""
        while True:
            try:
                await self._run_session()
            except asyncio.CancelledError:
                await self._teardown()
                raise
            except Exception as error:
                logger.error("[reactor] session error: %s", error)
                await self._teardown()
            logger.info("[reactor] reconnecting in %.0fs", RECONNECT_DELAY_S)
            await asyncio.sleep(RECONNECT_DELAY_S)

    async def _run_session(self) -> None:
        if self._config.local:
            # The SDK honours a non-default api_url in local mode, so a
            # runtime on another port (REACTOR_LOCAL_URL) works.
            reactor = Reactor(
                self._config.model, local=True, api_url=self._config.local_url
            )
        else:
            reactor = Reactor(self._config.model, api_key=self._config.api_key)
        disconnected = asyncio.Event()

        reactor.on("message", self._on_message)
        reactor.on_status(self._make_status_handler(disconnected))
        # Registered by wire name *before* connect: the SDK allows handler
        # registration ahead of the session declaring its tracks, whereas
        # querying `reactor.tracks` right after connect races that
        # declaration (an empty list on a slow session start).
        reactor.track("main_video").on_frame(self._on_video_frame)
        reactor.track("main_audio").on_frame(self._on_audio_frame)

        logger.info(
            "[reactor] connecting to %s (%s)...",
            self._config.model, "local" if self._config.local else "hosted",
        )
        await reactor.connect()
        logger.info(
            "[reactor] connected, session=%s status=%s",
            reactor.session_id, reactor.status,
        )

        self._reactor = reactor
        state = await asyncio.wait_for(self._raw_command(reactor, "get_state"), 30)
        if isinstance(state, dict) and "width" in state:
            self.state = state
        logger.info(
            "[reactor] canvas %dx%d, clip range %.3f-%.3fs, queue %d/%d",
            *self.canvas, self.min_seconds, self.max_seconds,
            self.queued, self.queue_capacity,
        )

        # Autoplay makes playback purely queue-driven: the oldest ready clip
        # starts whenever nothing is playing, so enqueue order is play order
        # and scene groups run back-to-back without a `play` per clip.
        await self._raw_command(reactor, "set_autoplay", {"enabled": True})

        self._first_state.set()
        self._ready.set()
        try:
            await disconnected.wait()
            logger.warning("[reactor] session disconnected")
        finally:
            await self._teardown()

    @staticmethod
    def _make_status_handler(disconnected: asyncio.Event):
        loop = asyncio.get_running_loop()

        def on_status(status: ReactorStatus) -> None:
            logger.info("[reactor] status: %s", status.value)
            if status == ReactorStatus.DISCONNECTED:
                loop.call_soon_threadsafe(disconnected.set)

        return on_status

    @staticmethod
    async def _raw_command(reactor: Reactor, command: str, data: dict | None = None) -> Any:
        return payload(await reactor.send_command(command, data or {}))

    async def _teardown(self) -> None:
        self._ready.clear()
        reactor, self._reactor = self._reactor, None
        if reactor is not None:
            try:
                await reactor.disconnect()
            except Exception:
                pass

    # ---------------------------------------------------------- media path

    def _on_video_frame(self, frame) -> None:
        if self._pacer is not None:
            self._pacer.submit_video(frame)

    def _on_audio_frame(self, frame, sample_rate=MODEL_SAMPLE_RATE) -> None:
        if sample_rate != MODEL_SAMPLE_RATE:
            logger.warning(
                "[reactor] audio at %dHz, expected %d — timing will drift",
                sample_rate, MODEL_SAMPLE_RATE,
            )
        if self._pacer is not None:
            self._pacer.submit_audio(frame)
