"""
Tests for the CC2 chamber-camera snapshot path.

Covers the CC2-only behaviour added to ElegooMjpegCamera: waiting for the
printer's MJPEG server to come up after the enable acknowledgement,
grabbing a single frame over a short-lived connection, serializing
overlapping grabs, debouncing the disable, and logging why a grab
returned None.

Everything here is mocked — no printer and no network are touched.
"""

import asyncio
import contextlib
from typing import Self
from unittest.mock import AsyncMock, MagicMock, patch

import custom_components.elegoo_printer.camera as camera_module
from custom_components.elegoo_printer.camera import ElegooMjpegCamera
from custom_components.elegoo_printer.sdcp.models.enums import ElegooVideoStatus

JPEG = b"\xff\xd8" + b"payload" + b"\xff\xd9"
STREAM_URL = "http://printer.invalid:8080/?action=stream"


def _run(coro):
    """Run an async coroutine to completion (fresh loop)."""
    asyncio.run(coro)


def _make_client(*, over_capacity: bool = False) -> MagicMock:
    """Build a mock CC2 printer client whose enable always succeeds."""
    client = MagicMock()
    client.is_connected = True
    client.printer_data = MagicMock()
    attrs = client.printer_data.attributes
    attrs.num_video_stream_connected = 2 if over_capacity else 0
    attrs.max_video_stream_allowed = 1
    video = MagicMock()
    video.status = ElegooVideoStatus.SUCCESS
    video.video_url = STREAM_URL
    client.printer_data.video = video
    client.get_printer_video = AsyncMock(return_value=video)
    client.set_printer_video_stream = AsyncMock()
    return client


def _cc2_camera(client: MagicMock, *, is_cc2: bool = True) -> ElegooMjpegCamera:
    """Build an ElegooMjpegCamera without running full entity init."""
    cam = object.__new__(ElegooMjpegCamera)
    cam.hass = MagicMock()
    cam.entity_id = "camera.chamber"
    cam._mjpeg_url = None
    cam._init_video_lifecycle(client)
    cam._is_cc2 = is_cc2
    return cam


class _FakeResponse:
    """Minimal aiohttp response stand-in usable as an async context manager."""

    def __init__(
        self,
        status: int = 200,
        body: bytes = JPEG,
        content_type: str = "multipart/x-mixed-replace",
    ) -> None:
        self.status = status
        self.headers = {"Content-Type": content_type}
        self._body = body
        self.content = MagicMock()
        self.content.iter_chunked = self._iter_chunked

    def _iter_chunked(self, _size: int):
        body = self._body

        async def gen():
            if body:
                yield body

        return gen()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class _FakeSession:
    """Session whose get() replays a queue of responses or errors."""

    def __init__(self, outcomes: list) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[str] = []

    def get(self, url: str, **_kwargs: object):
        self.calls.append(url)
        outcome = self._outcomes.pop(0) if self._outcomes else _FakeResponse(status=404)
        if isinstance(outcome, Exception):

            class _Raiser:
                async def __aenter__(self) -> Self:
                    raise outcome

                async def __aexit__(self, *_exc: object) -> bool:
                    return False

            return _Raiser()
        return outcome


def _patch_session(session: _FakeSession):
    """Patch the clientsession helper camera.py uses."""
    return patch.object(camera_module, "async_get_clientsession", return_value=session)


class TestCC2FrameFetch:
    """_fetch_single_frame extracts a frame and reports failures."""

    def test_returns_jpeg_from_stream(self) -> None:
        """A 200 response containing a JPEG yields the frame bytes."""

        async def run() -> None:
            cam = _cc2_camera(_make_client())
            session = _FakeSession([_FakeResponse(body=b"junk" + JPEG)])
            with _patch_session(session):
                image = await cam._fetch_single_frame(STREAM_URL)
            assert image == JPEG
            assert session.calls == [STREAM_URL]

        _run(run())

    def test_non_200_is_logged_with_status(self) -> None:
        """A 404 is reported with its status instead of failing silently."""

        async def run() -> None:
            cam = _cc2_camera(_make_client())
            session = _FakeSession([_FakeResponse(status=404, body=b"<html/>")])
            with (
                _patch_session(session),
                patch.object(camera_module.LOGGER, "warning") as warn,
            ):
                image = await cam._fetch_single_frame(STREAM_URL)
            assert image is None
            assert warn.call_count == 1
            assert "HTTP %d" in warn.call_args[0][0]

        _run(run())

    def test_body_without_jpeg_markers_is_logged(self) -> None:
        """A 200 with no JPEG in it is reported, not swallowed."""

        async def run() -> None:
            cam = _cc2_camera(_make_client())
            session = _FakeSession([_FakeResponse(body=b"not an image")])
            with (
                _patch_session(session),
                patch.object(camera_module.LOGGER, "warning") as warn,
            ):
                image = await cam._fetch_single_frame(STREAM_URL)
            assert image is None
            assert "no JPEG frame" in warn.call_args[0][0]

        _run(run())

    def test_connection_error_is_logged_with_type(self) -> None:
        """A transport error names the exception type and message."""

        async def run() -> None:
            cam = _cc2_camera(_make_client())
            session = _FakeSession([OSError("connection refused")])
            with (
                _patch_session(session),
                patch.object(camera_module.LOGGER, "warning") as warn,
            ):
                image = await cam._fetch_single_frame(STREAM_URL)
            assert image is None
            assert "connection refused" in str(warn.call_args)

        _run(run())


class TestStreamReadiness:
    """_wait_for_stream_ready retries until the server answers."""

    def test_ready_on_first_probe(self) -> None:
        """A server that is already up needs one probe and no sleeping."""

        async def run() -> None:
            cam = _cc2_camera(_make_client())
            session = _FakeSession([_FakeResponse(status=200)])
            with _patch_session(session):
                assert await cam._wait_for_stream_ready(STREAM_URL) is True
            assert len(session.calls) == 1

        _run(run())

    def test_retries_until_server_accepts(self) -> None:
        """Two refused connections then success — the grab still happens."""

        async def run() -> None:
            cam = _cc2_camera(_make_client())
            session = _FakeSession(
                [
                    OSError("connection refused"),
                    OSError("connection refused"),
                    _FakeResponse(status=200),
                ]
            )
            with (
                _patch_session(session),
                patch.object(camera_module.asyncio, "sleep", AsyncMock()),
            ):
                assert await cam._wait_for_stream_ready(STREAM_URL) is True
            assert len(session.calls) == 3

        _run(run())

    def test_gives_up_and_warns_after_timeout(self) -> None:
        """A server that never comes up is reported once at WARNING."""

        async def run() -> None:
            cam = _cc2_camera(_make_client())
            session = _FakeSession([OSError("refused")] * 50)
            with (
                _patch_session(session),
                patch.object(camera_module.asyncio, "sleep", AsyncMock()),
                patch.object(camera_module.LOGGER, "warning") as warn,
            ):
                assert await cam._wait_for_stream_ready(STREAM_URL) is False
            assert "not ready" in warn.call_args[0][0]

        _run(run())


class TestCC2CameraImage:
    """The CC2 async_camera_image path end to end."""

    def test_successful_grab_enables_and_debounces_disable(self) -> None:
        """A good grab returns bytes and does not disable immediately."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client)
            session = _FakeSession([_FakeResponse(status=200), _FakeResponse()])
            with _patch_session(session):
                image = await cam.async_camera_image()
            assert image == JPEG
            client.get_printer_video.assert_called_once_with(enable=True)
            # Disable is deferred, not sent inline.
            client.set_printer_video_stream.assert_not_called()
            assert cam._pending_disable_task is not None
            cam._cancel_pending_disable()

        _run(run())

    def test_debounced_disable_fires_after_delay(self) -> None:
        """Once the debounce window passes, the stream is disabled."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client)
            cam._stream_enabled = True
            with patch.object(camera_module, "DISABLE_DEBOUNCE_DELAY", 0):
                cam._schedule_disable()
                await cam._pending_disable_task
            client.set_printer_video_stream.assert_called_once_with(enable=False)
            assert cam._stream_enabled is False

        _run(run())

    def test_second_grab_inside_window_reuses_stream(self) -> None:
        """A grab during the debounce cancels the disable and re-enables nothing."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client)
            session = _FakeSession([_FakeResponse(status=200), _FakeResponse()] * 2)
            with _patch_session(session):
                await cam.async_camera_image()
                first_task = cam._pending_disable_task
                await cam.async_camera_image()
            # Let the cancellation actually land before inspecting the task.
            with contextlib.suppress(asyncio.CancelledError):
                await first_task
            assert first_task.cancelled()
            # Stream stayed enabled, so no second enable was needed.
            assert client.get_printer_video.call_count == 1
            client.set_printer_video_stream.assert_not_called()
            cam._cancel_pending_disable()

        _run(run())

    def test_reused_stream_skips_readiness_probe(self) -> None:
        """
        An already-enabled stream goes straight to the grab.

        The readiness probe costs a connection, which matters when the
        printer allows very few. It is only worth it right after an enable.
        """

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client)
            cam._mjpeg_url = STREAM_URL
            cam._stream_enabled = True
            session = _FakeSession([_FakeResponse()])
            with (
                _patch_session(session),
                patch.object(cam, "_wait_for_stream_ready", AsyncMock()) as ready,
            ):
                image = await cam.async_camera_image()
            assert image == JPEG
            ready.assert_not_called()
            # Exactly one connection: the frame grab itself.
            assert session.calls == [STREAM_URL]
            cam._cancel_pending_disable()

        _run(run())

    def test_first_grab_after_enable_waits_for_readiness(self) -> None:
        """A freshly enabled stream is probed before the grab."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client)
            session = _FakeSession([_FakeResponse()])
            with (
                _patch_session(session),
                patch.object(
                    cam, "_wait_for_stream_ready", AsyncMock(return_value=True)
                ) as ready,
            ):
                image = await cam.async_camera_image()
            assert image == JPEG
            ready.assert_awaited_once_with(STREAM_URL)
            cam._cancel_pending_disable()

        _run(run())

    def test_over_capacity_logs_counters(self) -> None:
        """The over-capacity path names both counters."""

        async def run() -> None:
            client = _make_client(over_capacity=True)
            cam = _cc2_camera(client)
            cam._mjpeg_url = STREAM_URL
            cam._stream_enabled = True
            with patch.object(camera_module.LOGGER, "warning") as warn:
                image = await cam.async_camera_image()
            assert image is None
            assert "num_video_stream_connected" in warn.call_args[0][0]
            assert warn.call_args[0][2] == 2
            assert warn.call_args[0][3] == 1

        _run(run())

    def test_no_url_is_logged(self) -> None:
        """A missing stream URL is reported rather than silently returning None."""

        async def run() -> None:
            client = _make_client()
            client.printer_data.video.video_url = ""
            cam = _cc2_camera(client)
            with patch.object(camera_module.LOGGER, "warning") as warn:
                image = await cam.async_camera_image()
            assert image is None
            assert "no stream URL" in warn.call_args[0][0]

        _run(run())

    def test_grabs_are_serialized(self) -> None:
        """Overlapping grabs cannot interleave enable/disable."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client)
            order: list[str] = []
            original = cam._fetch_single_frame

            async def tracked(url: str) -> bytes | None:
                order.append("start")
                await asyncio.sleep(0)
                order.append("end")
                return await original(url)

            cam._fetch_single_frame = tracked
            session = _FakeSession([_FakeResponse(status=200), _FakeResponse()] * 2)
            with _patch_session(session):
                await asyncio.gather(cam.async_camera_image(), cam.async_camera_image())
            # Never "start, start" — the lock kept the two grabs apart.
            assert order == ["start", "end", "start", "end"]
            cam._cancel_pending_disable()

        _run(run())

    def test_endpoint_probe_runs_once_after_failure(self) -> None:
        """The diagnostic probe fires on failure and only once per entity."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client)
            session = _FakeSession([_FakeResponse(status=404)] * 40)
            with (
                _patch_session(session),
                patch.object(camera_module.asyncio, "sleep", AsyncMock()),
                patch.object(
                    camera_module.MjpegCamera,
                    "async_camera_image",
                    AsyncMock(return_value=None),
                ),
                patch.object(camera_module.LOGGER, "warning") as warn,
            ):
                assert await cam.async_camera_image() is None
                assert cam._endpoints_probed is True
                probe_logs = [
                    c for c in warn.call_args_list if "endpoint probe" in c[0][0]
                ]
                assert len(probe_logs) == 1
                # A second failure must not re-probe.
                cam._last_grab_failure_log = 0.0
                await cam.async_camera_image()
                probe_logs = [
                    c for c in warn.call_args_list if "endpoint probe" in c[0][0]
                ]
                assert len(probe_logs) == 1
            cam._cancel_pending_disable()

        _run(run())


class TestNonCC2Unaffected:
    """SDCP/CC1 printers keep the original, non-CC2 behaviour."""

    def test_non_cc2_does_not_wait_or_probe(self) -> None:
        """A non-CC2 MJPEG camera goes straight to the base image path."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client, is_cc2=False)
            with (
                patch.object(
                    camera_module.MjpegCamera,
                    "async_camera_image",
                    AsyncMock(return_value=JPEG),
                ) as base,
                patch.object(cam, "_wait_for_stream_ready", AsyncMock()) as ready,
                patch.object(cam, "_fetch_single_frame", AsyncMock()) as fetch,
            ):
                image = await cam.async_camera_image()
            assert image == JPEG
            base.assert_awaited_once()
            ready.assert_not_called()
            fetch.assert_not_called()

        _run(run())

    def test_non_cc2_disables_inline_not_debounced(self) -> None:
        """The non-CC2 path still disables as soon as the last viewer leaves."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client, is_cc2=False)
            with patch.object(
                camera_module.MjpegCamera,
                "async_camera_image",
                AsyncMock(return_value=JPEG),
            ):
                await cam.async_camera_image()
            client.set_printer_video_stream.assert_called_once_with(enable=False)
            assert cam._pending_disable_task is None

        _run(run())

    def test_non_cc2_failure_is_still_logged(self) -> None:
        """A silent None from the base class is reported on every transport."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client, is_cc2=False)
            with (
                patch.object(
                    camera_module.MjpegCamera,
                    "async_camera_image",
                    AsyncMock(return_value=None),
                ),
                patch.object(camera_module.LOGGER, "warning") as warn,
            ):
                assert await cam.async_camera_image() is None
            assert "returned no image" in warn.call_args[0][0]

        _run(run())


class TestGrabFailureLogRateLimit:
    """_log_grab_failure warns once, then drops to debug."""

    def test_repeat_failures_are_rate_limited(self) -> None:
        """Only the first failure in the interval is a WARNING."""

        async def run() -> None:
            cam = _cc2_camera(_make_client())
            with (
                patch.object(camera_module.LOGGER, "warning") as warn,
                patch.object(camera_module.LOGGER, "debug") as debug,
            ):
                for _ in range(5):
                    cam._log_grab_failure("boom")
            assert warn.call_count == 1
            assert debug.call_count == 4

        _run(run())
