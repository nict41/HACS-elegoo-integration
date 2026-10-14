"""
Tests for the CC2 frame hub: one printer connection for every consumer.

The CC2 camera allows exactly one viewer and its camera server has been
crashed by two consumers competing for that slot. These tests pin down
that snapshots and live viewers share a single upstream connection, that
stale frames are never served, that reconnects back off without bursts of
video commands, and that the hub starts and stops on demand.

Everything here is mocked — no printer and no network are touched.
"""

import asyncio
import contextlib
import itertools
import time
from typing import Self
from unittest.mock import AsyncMock, MagicMock, patch

import custom_components.elegoo_printer.camera as camera_module
import custom_components.elegoo_printer.cc2.frame_hub as frame_hub_module
from custom_components.elegoo_printer.camera import ElegooMjpegCamera
from custom_components.elegoo_printer.cc2.frame_hub import (
    STATE_BACKOFF,
    CC2FrameHub,
    backoff_delay,
)
from custom_components.elegoo_printer.sdcp.models.enums import ElegooVideoStatus

STREAM_URL = "http://printer.invalid:8080/?action=stream"
FRAME_INTERVAL = 0.002  # seconds between frames from the fake printer


def _run(coro):
    """Run an async coroutine to completion (fresh loop)."""
    asyncio.run(coro)


def _jpeg(n: int) -> bytes:
    return b"\xff\xd8" + f"frame-{n}".encode() + b"\xff\xd9"


def _make_client(*, over_capacity: bool = False) -> MagicMock:
    """Build a mock CC2 printer client whose enable always succeeds."""
    client = MagicMock()
    client.is_connected = True
    client.printer_data = MagicMock()
    attrs = client.printer_data.attributes
    attrs.num_video_stream_connected = 1 if over_capacity else 0
    attrs.max_video_stream_allowed = 1
    attrs.camera_status = 1
    video = MagicMock()
    video.status = ElegooVideoStatus.SUCCESS
    video.video_url = STREAM_URL
    client.printer_data.video = video
    client.get_printer_video = AsyncMock(return_value=video)
    client.set_printer_video_stream = AsyncMock()
    return client


def _cc2_camera(client: MagicMock, *, passive: bool = False) -> ElegooMjpegCamera:
    """Build an ElegooMjpegCamera without running full entity init."""
    cam = object.__new__(ElegooMjpegCamera)
    cam.hass = MagicMock()
    cam.entity_id = "camera.chamber"
    cam._mjpeg_url = None
    cam._init_video_lifecycle(client)
    cam._is_cc2 = True
    cam._cc2_passive = passive
    # A hub that cannot stream reports via the port probe; keep it offline.
    cam._probe_camera_ports = AsyncMock()
    return cam


class _StreamResponse:
    """An MJPEG response that streams frames until told to stop."""

    def __init__(
        self,
        session: "_StreamSession",
        *,
        frames: int | None = None,
        then_stall: bool = False,
        status: int = 200,
    ) -> None:
        self._session = session
        self._frames = frames
        self._then_stall = then_stall
        self.status = status
        self.headers = {"Content-Type": "multipart/x-mixed-replace;boundary=b"}
        self.content = MagicMock()
        self.content.iter_chunked = self._iter_chunked

    def _iter_chunked(self, _size: int):
        async def gen():
            n = 0
            while self._frames is None or n < self._frames:
                await asyncio.sleep(FRAME_INTERVAL)
                part = b"--b\r\nContent-Type: image/jpeg\r\n\r\n" + _jpeg(n) + b"\r\n"
                # Split each part so frames also straddle chunk boundaries.
                yield part[:7]
                yield part[7:]
                n += 1
            if self._then_stall:
                await asyncio.Event().wait()

        return gen()

    async def __aenter__(self) -> Self:
        self._session.open += 1
        self._session.max_open = max(self._session.max_open, self._session.open)
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        self._session.open -= 1
        return False


class _Raiser:
    def __init__(self, err: Exception) -> None:
        self._err = err

    async def __aenter__(self) -> Self:
        raise self._err

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class _StreamSession:
    """
    Session whose get() opens a fake MJPEG stream (or fails).

    Tracks how many connections are open at once, which is the number the
    printer cares about.
    """

    def __init__(self, **response_kwargs: object) -> None:
        self._response_kwargs = response_kwargs
        self.error: Exception | None = None
        self.calls: list[str] = []
        self.open = 0
        self.max_open = 0

    def get(self, url: str, **_kwargs: object):
        self.calls.append(url)
        if self.error is not None:
            return _Raiser(self.error)
        return _StreamResponse(self, **self._response_kwargs)


def _patch_session(session: _StreamSession):
    return patch.object(
        frame_hub_module, "async_get_clientsession", return_value=session
    )


class _FakeTransport:
    def __init__(self) -> None:
        self.closing = False

    def is_closing(self) -> bool:
        return self.closing


class _FakeRequest:
    """Just enough of an aiohttp request for the CC2 stream handler."""

    def __init__(self, query: dict[str, str] | None = None) -> None:
        self.query = query or {}
        self.transport = _FakeTransport()

    def disconnect(self) -> None:
        self.transport.closing = True


class _FakeStreamResponse:
    """Stands in for web.StreamResponse; records what the viewer was sent."""

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.writes: list[bytes] = []
        self._request: _FakeRequest | None = None

    async def prepare(self, request: _FakeRequest) -> None:
        self._request = request

    async def write(self, data: bytes) -> None:
        if self._request is not None and self._request.transport.closing:
            raise ConnectionResetError
        self.writes.append(data)
        await asyncio.sleep(0)


def _patch_stream_response():
    return patch.object(camera_module.web, "StreamResponse", _FakeStreamResponse)


async def _until(predicate, limit: float = 3.0) -> None:
    """Poll until predicate() is true, failing the test after `limit` seconds."""
    async with asyncio.timeout(limit):
        while not predicate():  # noqa: ASYNC110 - polling hub state, no event to await
            await asyncio.sleep(0.005)


async def _start_viewer(cam, request: _FakeRequest) -> asyncio.Task:
    return asyncio.create_task(cam.handle_async_mjpeg_stream(request))


class TestSingleConnection:
    """However many consumers there are, the printer sees one connection."""

    def test_concurrent_snapshots_and_viewers_share_one_connection(self) -> None:
        """N snapshots + M viewers -> exactly one upstream session.get."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client)
            session = _StreamSession()
            requests = [_FakeRequest() for _ in range(3)]
            with _patch_session(session), _patch_stream_response():
                viewers = [await _start_viewer(cam, r) for r in requests]
                snaps = await asyncio.gather(
                    *(cam.async_camera_image() for _ in range(8))
                )
                hub = cam._frame_hub
                await _until(lambda: hub.frame_count >= 20)
                # A second wave of snapshots while streaming reuses the frame.
                snaps += await asyncio.gather(
                    *(cam.async_camera_image() for _ in range(8))
                )
                assert hub.subscribers == 3
                for request in requests:
                    request.disconnect()
                await asyncio.gather(*viewers)
                await cam._stop_frame_hub("test")

            assert all(s is not None and s.startswith(b"\xff\xd8") for s in snaps)
            assert len(session.calls) == 1
            assert session.max_open == 1
            assert hub.max_open_connections == 1
            assert hub.connects == 1
            # One ENABLE for the one connection.
            client.get_printer_video.assert_called_once_with(enable=True)

        _run(run())

    def test_viewers_receive_multipart_jpeg_frames(self) -> None:
        """Each viewer gets well-formed multipart parts with whole JPEGs."""

        async def run() -> None:
            cam = _cc2_camera(_make_client())
            session = _StreamSession()
            request = _FakeRequest()
            with _patch_session(session), _patch_stream_response():
                task = await _start_viewer(cam, request)
                await _until(
                    lambda: (
                        cam._frame_hub is not None and cam._frame_hub.frame_count >= 5
                    )
                )
                request.disconnect()
                response = await task
                await cam._stop_frame_hub("test")

            assert response.headers["Content-Type"] == (
                "multipart/x-mixed-replace;boundary=frame"
            )
            assert response.writes
            for part in response.writes:
                header, _, body = part.partition(b"\r\n\r\n")
                assert header.startswith(b"--frame\r\nContent-Type: image/jpeg")
                jpeg = body[:-2]
                assert f"Content-Length: {len(jpeg)}".encode() in header
                assert jpeg.startswith(b"\xff\xd8")
                assert jpeg.endswith(b"\xff\xd9")

        _run(run())

    def test_reconnects_never_overlap(self) -> None:
        """A stream that keeps dropping is reconnected one at a time."""

        async def run() -> None:
            cam = _cc2_camera(_make_client())
            session = _StreamSession(frames=2)
            hub = cam._get_frame_hub()

            async def no_wait(_delay: float) -> None:
                await asyncio.sleep(0)

            hub._backoff_sleep = no_wait
            with _patch_session(session):
                await cam.async_camera_image()
                await _until(lambda: len(session.calls) >= 10)
                await cam._stop_frame_hub("test")

            assert session.max_open == 1
            assert hub.max_open_connections == 1
            assert session.open == 0

        _run(run())

    def test_passive_hub_cycle_sends_no_video_commands(self) -> None:
        """Passive mode: start, stream, snapshot and stop with zero commands."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client, passive=True)
            cam._mjpeg_url = STREAM_URL
            session = _StreamSession()
            with (
                _patch_session(session),
                patch.object(camera_module, "DISABLE_DEBOUNCE_DELAY", 0),
            ):
                assert await cam.async_camera_image() is not None
                await cam._frame_hub.async_stop("test")
                if cam._pending_disable_task is not None:
                    await cam._pending_disable_task
            client.get_printer_video.assert_not_called()
            client.set_printer_video_stream.assert_not_called()

        _run(run())


class TestStaleFrames:
    """An old frame is never served as a snapshot."""

    def test_stalled_stream_returns_none(self) -> None:
        """Connected but no new frames: the old frame goes stale -> None."""

        async def run() -> None:
            cam = _cc2_camera(_make_client())
            session = _StreamSession(frames=1, then_stall=True)
            with (
                _patch_session(session),
                patch.object(frame_hub_module, "HUB_STALE_AFTER", 0.05),
                patch.object(camera_module, "FRAME_FETCH_TIMEOUT", 0.1),
            ):
                assert await cam.async_camera_image() == _jpeg(0)
                await asyncio.sleep(0.1)
                assert await cam.async_camera_image() is None
                await cam._stop_frame_hub("test")
            assert cam._stats["images_failed"] == 1

        _run(run())

    def test_backoff_returns_none_without_waiting(self) -> None:
        """While the hub is backing off no frame is coming, so don't wait."""

        async def run() -> None:
            cam = _cc2_camera(_make_client())
            session = _StreamSession(frames=1)
            hub = cam._get_frame_hub()

            async def park(_delay: float) -> None:
                await asyncio.Event().wait()

            hub._backoff_sleep = park
            with (
                _patch_session(session),
                patch.object(frame_hub_module, "HUB_STALE_AFTER", 0.05),
            ):
                assert await cam.async_camera_image() == _jpeg(0)
                await _until(lambda: hub.state == STATE_BACKOFF)
                await asyncio.sleep(0.1)
                started = time.monotonic()
                assert await cam.async_camera_image() is None
                assert time.monotonic() - started < 1.0
                await cam._stop_frame_hub("test")

        _run(run())

    def test_fresh_frame_is_served_from_memory(self) -> None:
        """A recent frame is returned as is; old frames are refused."""

        async def run() -> None:
            hub = CC2FrameHub(MagicMock(), "camera.x", prepare=AsyncMock())
            hub.latest_frame = b"\xff\xd8x\xff\xd9"
            hub.frame_time = asyncio.get_running_loop().time()
            assert hub.fresh_frame() == hub.latest_frame
            hub.frame_time -= frame_hub_module.HUB_STALE_AFTER + 1
            assert hub.fresh_frame() is None

        _run(run())


class TestBackoff:
    """Failed connections back off exponentially, one ENABLE per step."""

    def test_delays_grow_and_are_capped(self) -> None:
        """2, 4, 8 ... up to 120 s, never above the cap even with jitter."""
        with patch.object(frame_hub_module.random, "uniform", return_value=1.0):
            bases = [backoff_delay(n) for n in range(1, 11)]
        assert bases == [2, 4, 8, 16, 32, 64, 120, 120, 120, 120]
        with patch.object(frame_hub_module.random, "uniform", return_value=1.1):
            assert backoff_delay(50) == frame_hub_module.HUB_BACKOFF_MAX
        for n in range(1, 30):
            delay = backoff_delay(n)
            assert 0 < delay <= frame_hub_module.HUB_BACKOFF_MAX

    def test_failing_upstream_backs_off_with_one_enable_per_step(self) -> None:
        """A refusing camera gets one ENABLE and one connect per backoff step."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client)
            session = _StreamSession()
            session.error = OSError("connection refused")
            hub = cam._get_frame_hub()
            events: list[str] = []
            delays: list[float] = []
            steps = 10

            original_get = session.get

            def tracked_get(url: str, **kwargs: object):
                events.append("connect")
                return original_get(url, **kwargs)

            session.get = tracked_get
            enable = client.get_printer_video

            async def tracked_enable(**kwargs: object):
                events.append("enable")
                return await enable(**kwargs)

            client.get_printer_video = tracked_enable

            async def fake_sleep(delay: float) -> None:
                events.append("sleep")
                delays.append(delay)
                if len(delays) >= steps:
                    await asyncio.Event().wait()
                await asyncio.sleep(0)

            hub._backoff_sleep = fake_sleep
            with (
                _patch_session(session),
                patch.object(frame_hub_module.LOGGER, "warning") as warn,
            ):
                assert await cam.async_camera_image() is None
                await _until(lambda: len(delays) >= steps)
                await cam._stop_frame_hub("test")

            # Strictly growing until the cap, then pinned near it.
            assert all(a < b for a, b in itertools.pairwise(delays[:7]))
            assert all(d <= frame_hub_module.HUB_BACKOFF_MAX for d in delays)
            assert all(d >= 0.9 * frame_hub_module.HUB_BACKOFF_MAX for d in delays[6:])
            # Never a burst: each step is exactly enable -> connect -> sleep.
            assert events == ["enable", "connect", "sleep"] * steps
            assert enable.await_count == steps
            client.set_printer_video_stream.assert_not_called()
            assert hub.max_open_connections == 1
            # One WARNING for the whole streak, and one unreachable report.
            streak = [c for c in warn.call_args_list if "times in a row" in c[0][0]]
            assert len(streak) == 1
            cam._probe_camera_ports.assert_awaited_once()

        _run(run())

    def test_over_capacity_backs_off_without_connecting(self) -> None:
        """A slot held by someone else is left alone; the hub just waits."""

        async def run() -> None:
            client = _make_client(over_capacity=True)
            cam = _cc2_camera(client)
            session = _StreamSession()
            hub = cam._get_frame_hub()
            delays: list[float] = []

            async def fake_sleep(delay: float) -> None:
                delays.append(delay)
                if len(delays) >= 3:
                    await asyncio.Event().wait()

            hub._backoff_sleep = fake_sleep
            with _patch_session(session):
                assert await cam.async_camera_image() is None
                await _until(lambda: len(delays) >= 3)
                await cam._stop_frame_hub("test")
            assert session.calls == []
            client.get_printer_video.assert_not_called()

        _run(run())


class TestOnDemandLifecycle:
    """The hub runs only while someone wants frames."""

    def test_idle_timeout_stops_hub_and_restart_uses_one_connection(self) -> None:
        """Idle -> stop + debounced disable; next consumer -> one new connection."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client)
            session = _StreamSession()
            with (
                _patch_session(session),
                patch.object(frame_hub_module, "HUB_IDLE_TIMEOUT", 0.05),
                patch.object(camera_module, "DISABLE_DEBOUNCE_DELAY", 0),
            ):
                assert await cam.async_camera_image() is not None
                hub = cam._frame_hub
                assert hub.is_running
                await _until(lambda: cam._pending_disable_task is not None)
                assert not hub.is_running
                assert session.open == 0
                # The hub was the last viewer, so video is released.
                await cam._pending_disable_task
                client.set_printer_video_stream.assert_called_once_with(enable=False)
                assert cam._stream_enabled is False

                # The last frame is still fresh, so this snapshot may be
                # served before the restarted reader connects; wait for it.
                assert await cam.async_camera_image() is not None
                assert hub.is_running
                await _until(lambda: len(session.calls) == 2)
                assert session.max_open == 1
                await cam._stop_frame_hub("test")

        _run(run())

    def test_viewer_keeps_hub_alive_past_idle_timeout(self) -> None:
        """A connected viewer is demand; the idle timer waits for it to leave."""

        async def run() -> None:
            cam = _cc2_camera(_make_client())
            session = _StreamSession()
            request = _FakeRequest()
            with (
                _patch_session(session),
                _patch_stream_response(),
                patch.object(frame_hub_module, "HUB_IDLE_TIMEOUT", 0.05),
            ):
                task = await _start_viewer(cam, request)
                await _until(lambda: cam._frame_hub is not None)
                hub = cam._frame_hub
                await asyncio.sleep(0.2)
                assert hub.is_running
                request.disconnect()
                await task
                await _until(lambda: not hub.is_running)
            assert len(session.calls) == 1

        _run(run())

    def test_hub_counts_as_a_viewer(self) -> None:
        """While the hub runs, the idle watchdog leaves video enabled."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client)
            session = _StreamSession()
            with _patch_session(session):
                await cam.async_camera_image()
                cam._stale_slot_released = True
                assert cam._has_active_viewers()
                await cam._idle_watchdog_tick()
                client.set_printer_video_stream.assert_not_called()
                await cam._stop_frame_hub("test")
            assert not cam._has_active_viewers()

        _run(run())

    def test_home_assistant_stop_closes_hub_and_disables(self) -> None:
        """HA stopping closes the connection, disables video, and stays down."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client)
            session = _StreamSession()
            with _patch_session(session):
                await cam.async_camera_image()
                await cam._async_disable_on_stop(MagicMock())
                assert not cam._frame_hub.is_running
                assert session.open == 0
                client.set_printer_video_stream.assert_called_once_with(enable=False)
                # Nothing restarts it on the way out.
                assert await cam.async_camera_image() is None
            assert len(session.calls) == 1
            assert cam._pending_disable_task is None

        _run(run())

    def test_entity_removal_stops_hub(self) -> None:
        """async_will_remove_from_hass's cleanup closes the hub's connection."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client)
            session = _StreamSession()
            with _patch_session(session):
                await cam.async_camera_image()
                await cam._cleanup_video_lifecycle()
            assert not cam._frame_hub.is_running
            assert session.open == 0
            assert cam._pending_disable_task is None
            client.set_printer_video_stream.assert_called_once_with(enable=False)

        _run(run())


class TestViewerSubscriptions:
    """Viewers subscribe for exactly as long as they are connected."""

    def test_disconnect_mid_stream_frees_subscription(self) -> None:
        """A write to a closed client ends the viewer and drops its count."""

        async def run() -> None:
            cam = _cc2_camera(_make_client())
            session = _StreamSession()
            first, second = _FakeRequest(), _FakeRequest()
            with _patch_session(session), _patch_stream_response():
                task_a = await _start_viewer(cam, first)
                task_b = await _start_viewer(cam, second)
                await _until(lambda: cam._frame_hub is not None)
                hub = cam._frame_hub
                await _until(lambda: hub.subscribers == 2 and hub.frame_count > 3)
                first.disconnect()
                await task_a
                assert hub.subscribers == 1
                second.disconnect()
                await task_b
                assert hub.subscribers == 0
                await cam._stop_frame_hub("test")

        _run(run())

    def test_cancelled_viewer_frees_subscription(self) -> None:
        """A handler cancelled by aiohttp still releases its subscription."""

        async def run() -> None:
            cam = _cc2_camera(_make_client())
            session = _StreamSession()
            with _patch_session(session), _patch_stream_response():
                task = await _start_viewer(cam, _FakeRequest())
                await _until(lambda: cam._frame_hub is not None)
                hub = cam._frame_hub
                await _until(lambda: hub.subscribers == 1)
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                assert hub.subscribers == 0
                await cam._stop_frame_hub("test")

        _run(run())

    def test_viewer_with_no_frames_notices_disconnect(self) -> None:
        """While the camera is down, a gone client still ends its viewer."""

        async def run() -> None:
            cam = _cc2_camera(_make_client())
            session = _StreamSession()
            session.error = OSError("refused")
            request = _FakeRequest()
            with (
                _patch_session(session),
                _patch_stream_response(),
                patch.object(camera_module, "CC2_STREAM_WAIT", 0.01),
            ):
                task = await _start_viewer(cam, request)
                await _until(lambda: cam._frame_hub is not None)
                hub = cam._frame_hub
                await _until(lambda: hub.subscribers == 1)
                request.disconnect()
                await task
                assert hub.subscribers == 0
                await cam._stop_frame_hub("test")

        _run(run())

    def test_fps_cap_throttles_one_viewer_only(self) -> None:
        """?fps= limits that viewer; a native viewer still gets every frame."""

        async def run() -> None:
            cam = _cc2_camera(_make_client())
            session = _StreamSession()
            slow, fast = _FakeRequest({"fps": "10"}), _FakeRequest()
            with _patch_session(session), _patch_stream_response():
                slow_task = await _start_viewer(cam, slow)
                fast_task = await _start_viewer(cam, fast)
                await asyncio.sleep(0.35)
                slow.disconnect()
                fast.disconnect()
                slow_resp = await slow_task
                fast_resp = await fast_task
                await cam._stop_frame_hub("test")
            # 10 fps for ~0.35 s is about 4 frames.
            assert 1 <= len(slow_resp.writes) <= 6
            assert len(fast_resp.writes) > 2 * len(slow_resp.writes)

        _run(run())

    def test_fps_parsing_is_clamped(self) -> None:
        """fps is clamped to 1-20; junk means native rate."""
        parse = ElegooMjpegCamera._parse_fps
        assert parse(None) is None
        assert parse("") is None
        assert parse("abc") is None
        assert parse("nan") is None
        assert parse("inf") is None
        assert parse("0") == 1.0
        assert parse("-5") == 1.0
        assert parse("5") == 5.0
        assert parse("2.5") == 2.5
        assert parse("1000") == 20.0


class TestStreamSourceAndStats:
    """No second path to the printer, and the hub shows up in the heartbeat."""

    def test_stream_source_is_none_on_cc2(self) -> None:
        """No stream source, so ffmpeg/go2rtc never open their own connection."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client)
            assert await cam.stream_source() is None
            client.get_printer_video.assert_not_called()
            assert cam._frame_hub is None

        _run(run())

    def test_activity_summary_includes_hub_stats(self) -> None:
        """Subscribers, frames, reconnects and frame age ride on the heartbeat."""

        async def run() -> None:
            cam = _cc2_camera(_make_client())
            session = _StreamSession()
            with (
                _patch_session(session),
                patch.object(camera_module.LOGGER, "info") as info,
            ):
                await cam.async_camera_image()
                cam._log_activity_summary()
                await cam._stop_frame_hub("test")
            line = next(c for c in info.call_args_list if "activity for" in c[0][0])
            hub_stats = line[0][-1]
            for key in ("subscribers=", "frames=", "reconnects=", "last_frame_age="):
                assert key in hub_stats

        _run(run())

    def test_hub_lifecycle_is_logged_with_marker(self) -> None:
        """Start, connect, disconnect and stop are INFO lines with CC2CAM."""

        async def run() -> None:
            cam = _cc2_camera(_make_client())
            session = _StreamSession()
            with (
                _patch_session(session),
                patch.object(frame_hub_module.LOGGER, "info") as info,
            ):
                await cam.async_camera_image()
                await cam._stop_frame_hub("test")
            lines = [c[0][0] for c in info.call_args_list]
            for needle in ("starting", "connected to", "disconnected", "stopped"):
                assert any(needle in line and "CC2CAM" in line for line in lines)

        _run(run())


class TestFrameParsing:
    """SOI/EOI splitting copes with junk and split markers."""

    def test_extracts_frames_across_chunk_boundaries(self) -> None:
        """Markers split between chunks still yield whole frames."""

        async def run() -> None:
            hub = CC2FrameHub(MagicMock(), "camera.x", prepare=AsyncMock())
            stream = b"junk" + _jpeg(1) + b"--b\r\n" + _jpeg(2) + b"\r\n"
            buf = bytearray()
            for i in range(len(stream)):
                buf += stream[i : i + 1]
                await hub._extract_frames(buf)
            assert hub.frame_count == 2
            assert hub.latest_frame == _jpeg(2)
            assert len(buf) <= 2

        _run(run())

    def test_runaway_buffer_without_frame_disconnects(self) -> None:
        """A stream with no JPEG in it is dropped at FRAME_FETCH_MAX_BYTES."""

        async def run() -> None:
            hub = CC2FrameHub(MagicMock(), "camera.x", prepare=AsyncMock())
            resp = MagicMock()

            def chunks(_size: int):
                async def gen():
                    while True:
                        yield b"\xff\xd8" + b"\x00" * 1024

                return gen()

            resp.content.iter_chunked = chunks
            with patch.object(frame_hub_module, "FRAME_FETCH_MAX_BYTES", 8192):
                reason = await hub._read_frames(resp)
            assert "no complete JPEG" in reason
            assert hub.frame_count == 0

        _run(run())
