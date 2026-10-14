"""
Single-connection MJPEG frame hub for the CC2 chamber camera.

The CC2 serves MJPEG on :8080 and allows exactly one viewer
(max_video_stream_allowed=1). Its camera server is fragile: a live view
competing with a snapshot poll for that one slot has crashed it outright,
leaving :8080 refusing connections until the printer was restarted.

CC2FrameHub makes the printer see one connection however many consumers
there are. It holds a single upstream MJPEG connection, keeps the latest
JPEG, and hands that frame to every snapshot and every live viewer. It
runs on demand: the first consumer starts it, and it stops once nobody
has watched or asked for a snapshot for HUB_IDLE_TIMEOUT seconds.

When the connection fails it backs off exponentially (HUB_BACKOFF_INITIAL
doubling up to HUB_BACKOFF_MAX, with jitter). Before each attempt it asks
its owner to get the printer ready, which is where the one video ENABLE
per backoff step is sent. It never retries in a tight loop.
"""

import asyncio
import contextlib
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from http import HTTPStatus
from typing import TYPE_CHECKING

import aiohttp
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from custom_components.elegoo_printer.const import LOGGER

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

# Every CC2 camera diagnostic line carries this marker so a whole session's
# timeline can be pulled out of a Home Assistant log with one grep.
LOG_MARKER = "CC2CAM"

HUB_IDLE_TIMEOUT = 60.0  # stop this long after the last viewer/snapshot
HUB_IDLE_CHECK_INTERVAL = 5.0  # how often a hub with viewers rechecks idleness
HUB_STALE_AFTER = 5.0  # a frame older than this is never served as a snapshot
HUB_BACKOFF_INITIAL = 2.0  # first reconnect delay
HUB_BACKOFF_MAX = 120.0  # reconnect delay ceiling
HUB_BACKOFF_JITTER = 0.1  # +/- fraction applied to each delay
HUB_WARN_AFTER_FAILURES = 5  # consecutive failures before one WARNING
HUB_STABLE_SESSION = 30.0  # a session this long resets the backoff
HUB_CONNECT_TIMEOUT = 10.0  # connect + response headers
HUB_READ_TIMEOUT = 10.0  # no bytes for this long means the stream has stalled
# No COMPLETE frame for this long also means it has stalled. Needed on top of
# HUB_READ_TIMEOUT because a half-started camera server can answer HTTP 200 and
# then trickle bytes that never form a JPEG: every chunk resets the no-bytes
# timer, and _extract_frames discards the junk, so the size guard never fills.
# Applies before the first frame of a session and between frames.
HUB_FRAME_TIMEOUT = 15.0
FRAME_FETCH_MAX_BYTES = 4 * 1024 * 1024  # give up on a buffer with no frame in it
BUFFER_SIZE = 102400  # matches homeassistant.components.mjpeg.camera

JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"

STATE_STOPPED = "stopped"
STATE_CONNECTING = "connecting"
STATE_STREAMING = "streaming"
STATE_BACKOFF = "backoff"


def backoff_delay(failures: int) -> float:
    """
    Return the delay before reconnect attempt number `failures`.

    2 s, 4 s, 8 s ... capped at HUB_BACKOFF_MAX, each with +/- jitter so a
    restart of Home Assistant does not line every retry up on the second.
    The jittered value is clamped to the cap as well.
    """
    exponent = min(max(failures, 1) - 1, 16)
    base = min(HUB_BACKOFF_INITIAL * 2**exponent, HUB_BACKOFF_MAX)
    jitter = random.uniform(1 - HUB_BACKOFF_JITTER, 1 + HUB_BACKOFF_JITTER)  # noqa: S311
    return min(base * jitter, HUB_BACKOFF_MAX)


class CC2FrameHub:
    """
    Fan one upstream MJPEG connection out to any number of consumers.

    Consumers either take a snapshot (async_snapshot) or subscribe as a
    live viewer (async_subscribe + async_next_frame). Neither opens a
    connection of its own: the single reader task is the only thing that
    ever talks to the camera.
    """

    def __init__(
        self,
        hass: "HomeAssistant",
        name: str,
        *,
        prepare: Callable[..., Awaitable[str | None]],
        on_stopped: Callable[[], None] | None = None,
        on_unreachable: Callable[[str, int, str], Awaitable[None]] | None = None,
    ) -> None:
        """
        Initialize the hub. Nothing is started until a consumer appears.

        Arguments:
            hass: The Home Assistant instance (for the shared client session).
            name: Name used in log lines, normally the camera's entity_id.
            prepare: Awaited before every connection attempt with
                `after_failure=` keyword. Gets the printer ready (at most one
                video ENABLE per call) and returns the stream URL, or None
                if no attempt should be made this step.
            on_stopped: Called after the hub has fully stopped.
            on_unreachable: Awaited once per failure streak when it reaches
                HUB_WARN_AFTER_FAILURES, with (url, attempts, last_error).

        """
        self._hass = hass
        self.name = name
        self._prepare = prepare
        self._on_stopped = on_stopped
        self._on_unreachable = on_unreachable
        # Seam for tests: only backoff waits go through this.
        self._backoff_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

        self.latest_frame: bytes | None = None
        self.frame_time: float | None = None  # loop.time(), i.e. monotonic
        self.frame_count = 0
        self.state = STATE_STOPPED
        self.subscribers = 0
        self.connects = 0
        self.reconnects = 0
        self.consecutive_failures = 0
        # Single-flight bookkeeping, checked by tests.
        self.open_connections = 0
        self.max_open_connections = 0

        self._cond = asyncio.Condition()
        self._connect_lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._idle_task: asyncio.Task | None = None
        self._last_demand = 0.0
        self._session_frames = 0
        self._warned = False
        self._stopping = False
        self._restart_after_stop = False
        self._closed = False

    # -- state ---------------------------------------------------------------

    @staticmethod
    def _now() -> float:
        return asyncio.get_running_loop().time()

    @property
    def is_running(self) -> bool:
        """Return True while the reader task is alive (streaming or backing off)."""
        return self._task is not None and not self._task.done()

    def frame_age(self) -> float | None:
        """Seconds since the latest frame arrived, or None if there is none."""
        if self.frame_time is None:
            return None
        return self._now() - self.frame_time

    def fresh_frame(self) -> bytes | None:
        """Return the latest frame unless it is older than HUB_STALE_AFTER."""
        age = self.frame_age()
        if age is None or age > HUB_STALE_AFTER:
            return None
        return self.latest_frame

    def describe(self) -> str:
        """One-line summary for the camera's activity log."""
        age = self.frame_age()
        return (
            f"state={self.state} subscribers={self.subscribers} "
            f"frames={self.frame_count} connects={self.connects} "
            f"reconnects={self.reconnects} "
            f"consecutive_failures={self.consecutive_failures} "
            f"last_frame_age={'n/a' if age is None else f'{age:.1f}s'}"
        )

    # -- consumers -------------------------------------------------------------

    def ensure_started(self) -> None:
        """
        Start the reader if it is not running. Idempotent and synchronous.

        Being synchronous is what makes this single-flight: there is no
        await between the running check and creating the task, so two
        consumers arriving together cannot both start a reader.
        """
        if self._closed:
            return
        if self._stopping:
            # A stop is in progress; start again as soon as it finishes.
            self._restart_after_stop = True
            return
        if self.is_running:
            return
        loop = asyncio.get_running_loop()
        self._last_demand = loop.time()
        self.consecutive_failures = 0
        self._warned = False
        self.state = STATE_CONNECTING
        LOGGER.info(
            LOG_MARKER + " frame hub for %s starting (subscribers=%d)",
            self.name,
            self.subscribers,
        )
        self._task = loop.create_task(self._run())
        self._idle_task = loop.create_task(self._idle_watch())

    async def async_snapshot(self, max_wait: float) -> bytes | None:
        """
        Return a fresh frame, starting the hub if needed.

        Serves the latest frame if it is younger than HUB_STALE_AFTER.
        Otherwise waits up to `max_wait` seconds for the next one, unless the hub is
        backing off, in which case no frame is coming and None is returned
        at once. An old frame is never returned.
        """
        if self._closed:
            return None
        self._last_demand = self._now()
        self.ensure_started()
        frame = self.fresh_frame()
        if frame is not None or not self.is_running or self.state == STATE_BACKOFF:
            return frame
        got = await self._wait_for_frame(
            self.frame_count, max_wait, give_up_on_backoff=True
        )
        return got[1] if got is not None else None

    @contextlib.asynccontextmanager
    async def async_subscribe(self) -> AsyncIterator["CC2FrameHub"]:
        """Register a live viewer for the duration of the context."""
        self.subscribers += 1
        LOGGER.info(
            LOG_MARKER + " frame hub for %s: viewer joined (subscribers=%d)",
            self.name,
            self.subscribers,
        )
        self.ensure_started()
        try:
            yield self
        finally:
            self.subscribers = max(0, self.subscribers - 1)
            # The idle grace period starts when the last viewer leaves.
            self._last_demand = self._now()
            LOGGER.info(
                LOG_MARKER + " frame hub for %s: viewer left (subscribers=%d)",
                self.name,
                self.subscribers,
            )

    async def async_next_frame(
        self, after: int, max_wait: float
    ) -> tuple[int, bytes] | None:
        """
        Wait for a frame newer than frame number `after`.

        Returns (frame_count, frame) for the newest frame - frames that
        arrived in between are skipped, which is how slow viewers are
        throttled without buffering - or None on timeout.
        """
        return await self._wait_for_frame(after, max_wait, give_up_on_backoff=False)

    async def _wait_for_frame(
        self, after: int, max_wait: float, *, give_up_on_backoff: bool
    ) -> tuple[int, bytes] | None:
        def ready() -> bool:
            if self.frame_count > after:
                return True
            return give_up_on_backoff and self.state in (STATE_BACKOFF, STATE_STOPPED)

        async with self._cond:
            try:
                async with asyncio.timeout(max_wait):
                    await self._cond.wait_for(ready)
            except TimeoutError:
                return None
            if self.frame_count > after and self.latest_frame is not None:
                return self.frame_count, self.latest_frame
            return None

    # -- lifecycle -------------------------------------------------------------

    async def async_stop(self, reason: str, *, final: bool = False) -> None:
        """
        Stop the reader and close the upstream connection.

        Arguments:
            reason: Logged with the stop.
            final: The owner is going away (entity removed, HA stopping);
                the hub will not start again.

        """
        if final:
            self._closed = True
            self._restart_after_stop = False
        if self._stopping or (self._task is None and self._idle_task is None):
            return
        self._stopping = True
        current = asyncio.current_task()
        tasks = [
            task
            for task in (self._task, self._idle_task)
            if task is not None and task is not current
        ]
        try:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        finally:
            self._task = None
            self._idle_task = None
            self._stopping = False
            self.state = STATE_STOPPED
            async with self._cond:
                self._cond.notify_all()
        LOGGER.info(
            LOG_MARKER + " frame hub for %s stopped (%s): %s",
            self.name,
            reason,
            self.describe(),
        )
        if self._on_stopped is not None:
            self._on_stopped()
        if self._restart_after_stop and not self._closed:
            self._restart_after_stop = False
            self.ensure_started()

    async def _idle_watch(self) -> None:
        """Stop the hub once nobody has wanted a frame for HUB_IDLE_TIMEOUT."""
        while True:
            if self.subscribers > 0:
                await asyncio.sleep(min(HUB_IDLE_TIMEOUT, HUB_IDLE_CHECK_INTERVAL))
                continue
            idle_for = self._now() - self._last_demand
            if idle_for >= HUB_IDLE_TIMEOUT:
                await self.async_stop(f"idle for {idle_for:.0f}s")
                return
            await asyncio.sleep(HUB_IDLE_TIMEOUT - idle_for)

    # -- reader ----------------------------------------------------------------

    async def _run(self) -> None:
        """Connect, read until the stream ends, back off, repeat."""
        after_failure = False
        while True:
            self.state = STATE_CONNECTING
            url: str | None = None
            frames, duration, error = 0, 0.0, "no stream URL"
            try:
                url = await self._prepare(after_failure=after_failure)
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001
                error = f"preparing the printer failed: {type(err).__name__}: {err}"
            if url:
                frames, duration, error = await self._stream_once(url)

            if frames and duration >= HUB_STABLE_SESSION:
                # A healthy session that ended is a fresh start, not a streak.
                self.consecutive_failures = 0
                self._warned = False
            self.consecutive_failures += 1
            delay = backoff_delay(self.consecutive_failures)
            self.state = STATE_BACKOFF
            async with self._cond:
                self._cond.notify_all()
            LOGGER.info(
                LOG_MARKER + " frame hub for %s: %s; reconnecting in %.1fs "
                "(consecutive failures=%d, subscribers=%d)",
                self.name,
                error,
                delay,
                self.consecutive_failures,
                self.subscribers,
            )
            if (
                self.consecutive_failures >= HUB_WARN_AFTER_FAILURES
                and not self._warned
            ):
                self._warned = True
                LOGGER.warning(
                    LOG_MARKER + " frame hub for %s has failed %d times in a row "
                    "(last error: %s). Staying in slow backoff, at most one "
                    "video enable every %.0fs, until the camera answers.",
                    self.name,
                    self.consecutive_failures,
                    error,
                    HUB_BACKOFF_MAX,
                )
                if self._on_unreachable is not None and url:
                    with contextlib.suppress(Exception):
                        await self._on_unreachable(
                            url, self.consecutive_failures, error
                        )
            await self._backoff_sleep(delay)
            self.reconnects += 1
            after_failure = True

    async def _stream_once(self, url: str) -> tuple[int, float, str]:
        """
        Hold one connection until it ends.

        Returns:
            (frames received, seconds connected, why it ended).

        """
        async with self._connect_lock:
            if self.open_connections:
                # Unreachable while the lock is held; kept as a hard guard.
                return 0, 0.0, "single-flight guard refused a second connection"
            self.open_connections += 1
            self.max_open_connections = max(
                self.max_open_connections, self.open_connections
            )
            self.connects += 1
            self._session_frames = 0
            started = self._now()
            reason = "hub stopped"
            try:
                reason = await self._consume(url)
            finally:
                self.open_connections -= 1
                duration = self._now() - started
                LOGGER.info(
                    LOG_MARKER + " frame hub for %s disconnected from %s after "
                    "%.1fs and %d frames: %s",
                    self.name,
                    url,
                    duration,
                    self._session_frames,
                    reason,
                )
            return self._session_frames, duration, reason

    async def _consume(self, url: str) -> str:
        """Read the multipart stream, publishing each JPEG. Returns why it ended."""
        session = async_get_clientsession(self._hass)
        timeout = aiohttp.ClientTimeout(
            total=None, sock_connect=HUB_CONNECT_TIMEOUT, sock_read=HUB_READ_TIMEOUT
        )
        try:
            async with contextlib.AsyncExitStack() as stack:
                async with asyncio.timeout(HUB_CONNECT_TIMEOUT):
                    resp = await stack.enter_async_context(
                        session.get(url, timeout=timeout)
                    )
                content_type = resp.headers.get("Content-Type", "no content-type")
                if resp.status != HTTPStatus.OK:
                    return f"HTTP {resp.status} ({content_type})"
                LOGGER.info(
                    LOG_MARKER + " frame hub for %s connected to %s (HTTP %d, %s)",
                    self.name,
                    url,
                    resp.status,
                    content_type,
                )
                self.state = STATE_STREAMING
                return await self._read_frames(resp)
        except TimeoutError:
            return "timed out"
        except (aiohttp.ClientError, OSError) as err:
            return f"{type(err).__name__}: {err}"

    async def _read_frames(self, resp: aiohttp.ClientResponse) -> str:
        """
        Split the byte stream on SOI/EOI markers, like the old single-frame grab.

        Two independent stall limits end the session: no bytes at all for
        HUB_READ_TIMEOUT, and no complete frame for HUB_FRAME_TIMEOUT. The
        second is what catches a server that keeps talking without ever
        sending a picture.
        """
        chunks = resp.content.iter_chunked(BUFFER_SIZE)
        buf = bytearray()
        last_frame_at = self._now()
        bytes_since_frame = 0
        while True:
            frame_deadline = last_frame_at + HUB_FRAME_TIMEOUT
            wait = min(HUB_READ_TIMEOUT, frame_deadline - self._now())
            if wait <= 0:
                return self._no_frame_reason(bytes_since_frame)
            try:
                async with asyncio.timeout(wait):
                    chunk = await anext(chunks)
            except StopAsyncIteration:
                return "printer closed the stream"
            except TimeoutError:
                if self._now() >= frame_deadline:
                    return self._no_frame_reason(bytes_since_frame)
                return f"no data for {HUB_READ_TIMEOUT:.0f}s"
            buf += chunk
            bytes_since_frame += len(chunk)
            if await self._extract_frames(buf):
                last_frame_at = self._now()
                bytes_since_frame = 0
            if len(buf) > FRAME_FETCH_MAX_BYTES:
                return f"no complete JPEG in {len(buf)} bytes"

    @staticmethod
    def _no_frame_reason(bytes_since_frame: int) -> str:
        # The byte count tells silence (0) apart from a trickle of non-JPEG data.
        return (
            f"no complete frame for {HUB_FRAME_TIMEOUT:.0f}s "
            f"({bytes_since_frame} bytes received without one)"
        )

    async def _extract_frames(self, buf: bytearray) -> int:
        """
        Publish every complete JPEG in `buf` and drop the consumed bytes.

        Returns:
            How many frames were published.

        """
        published = 0
        while True:
            start = buf.find(JPEG_SOI)
            if start == -1:
                # Keep the last byte: it may be the 0xff of a split marker.
                del buf[:-1]
                return published
            end = buf.find(JPEG_EOI, start + 2)
            if end == -1:
                del buf[:start]
                return published
            await self._publish(bytes(buf[start : end + 2]))
            published += 1
            del buf[: end + 2]

    async def _publish(self, frame: bytes) -> None:
        async with self._cond:
            self.latest_frame = frame
            self.frame_time = self._now()
            self.frame_count += 1
            self._session_frames += 1
            self._cond.notify_all()
