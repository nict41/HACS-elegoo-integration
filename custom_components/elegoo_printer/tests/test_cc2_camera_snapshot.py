"""
Tests for the CC2 chamber-camera snapshot path.

Covers the CC2-only behaviour of ElegooMjpegCamera around snapshots:
debouncing the disable, never leaving a video slot occupied, logging why
a grab returned None, and leaving the non-CC2 path alone. The frame hub
that now serves CC2 snapshots is covered in test_cc2_frame_hub.py.

Everything here is mocked — no printer and no network are touched.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import custom_components.elegoo_printer.camera as camera_module
import custom_components.elegoo_printer.cc2.frame_hub as frame_hub_module
from custom_components.elegoo_printer.camera import ElegooMjpegCamera
from custom_components.elegoo_printer.sdcp.models.enums import ElegooVideoStatus

JPEG = b"\xff\xd8" + b"payload" + b"\xff\xd9"
STREAM_URL = "http://printer.invalid:8080/?action=stream"
REFUSED = "refused"


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


def _patch_session(session: MagicMock):
    """Patch the clientsession helper the frame hub uses."""
    return patch.object(
        frame_hub_module, "async_get_clientsession", return_value=session
    )


class TestCC2CameraImage:
    """The CC2 async_camera_image path, which is served by the frame hub."""

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

    def test_over_capacity_logs_counters(self) -> None:
        """The over-capacity path names both counters and opens nothing."""

        async def run() -> None:
            client = _make_client(over_capacity=True)
            cam = _cc2_camera(client)
            cam._mjpeg_url = STREAM_URL
            cam._stream_enabled = True
            session = MagicMock()
            with (
                _patch_session(session),
                patch.object(camera_module.LOGGER, "warning") as warn,
            ):
                image = await cam.async_camera_image()
                await cam._stop_frame_hub("test")
            assert image is None
            # The printer says its one slot is taken: no connection is made.
            session.get.assert_not_called()
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
            session = MagicMock()
            with (
                _patch_session(session),
                patch.object(camera_module.LOGGER, "warning") as warn,
            ):
                image = await cam.async_camera_image()
                await cam._stop_frame_hub("test")
            assert image is None
            session.get.assert_not_called()
            assert "no stream URL" in warn.call_args[0][0]

        _run(run())


class TestUnreachableCameraDiagnostics:
    """What gets reported when nothing accepts the connection."""

    def test_port_probe_reports_open_and_closed_ports(self) -> None:
        """The probe says which ports listen, using bare TCP connects."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client)

            async def fake_open(_host: str, port: int):
                if port != 8081:
                    raise ConnectionRefusedError(REFUSED)
                writer = MagicMock()
                writer.close = MagicMock()
                writer.wait_closed = AsyncMock()
                return MagicMock(), writer

            with (
                patch.object(camera_module.asyncio, "open_connection", fake_open),
                patch.object(camera_module.LOGGER, "warning") as warn,
            ):
                await cam._probe_camera_ports(STREAM_URL)

            message = str(warn.call_args)
            assert "8081: open" in message
            assert "8080: ConnectionRefusedError" in message

        _run(run())

    def test_port_probe_runs_once_per_entity(self) -> None:
        """The probe does not repeat on every failed snapshot."""

        async def run() -> None:
            cam = _cc2_camera(_make_client())

            async def fake_open(_host: str, _port: int):
                raise ConnectionRefusedError(REFUSED)

            with (
                patch.object(camera_module.asyncio, "open_connection", fake_open),
                patch.object(camera_module.LOGGER, "warning") as warn,
            ):
                await cam._probe_camera_ports(STREAM_URL)
                await cam._probe_camera_ports(STREAM_URL)
            assert warn.call_count == 1

        _run(run())

    def test_camera_status_disconnected_is_surfaced(self) -> None:
        """A printer reporting no camera attached says so in the warning."""

        async def run() -> None:
            client = _make_client()
            client.printer_data.attributes.camera_status = 0
            cam = _cc2_camera(client)
            with (
                patch.object(cam, "_probe_camera_ports", AsyncMock()),
                patch.object(camera_module.LOGGER, "warning") as warn,
            ):
                await cam._report_unreachable_camera(STREAM_URL, 8, "refused")
            # args: (format, entity_id, url, attempts,
            #        last_error, camera_status, num_connected, max_allowed)
            assert warn.call_args[0][4] == "refused"
            assert warn.call_args[0][5] == 0

        _run(run())


class TestSlotPrevention:
    """Never leave a video slot occupied."""

    def test_startup_sends_unconditional_disable(self) -> None:
        """
        A slot leaked by a previous run is released when the entity is added.

        _stream_enabled starts False, so the normal disable path would
        never touch a stream the printer is still holding open.
        """

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client)
            assert cam._stream_enabled is False
            released = await cam._release_stale_stream()
            assert released is True
            client.set_printer_video_stream.assert_called_once_with(enable=False)

        _run(run())

    def test_startup_release_retried_while_disconnected(self) -> None:
        """A printer that is not reachable yet is retried by the watchdog."""

        async def run() -> None:
            client = _make_client()
            client.is_connected = False
            cam = _cc2_camera(client)
            assert await cam._release_stale_stream() is False
            client.set_printer_video_stream.assert_not_called()

            # Watchdog picks it up once the printer is back.
            client.is_connected = True
            await cam._idle_watchdog_tick()
            assert cam._stale_slot_released is True
            client.set_printer_video_stream.assert_called_once_with(enable=False)

        _run(run())

    def test_watchdog_does_not_re_release_once_done(self) -> None:
        """The startup release happens once, not on every watchdog pass."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client)
            cam._stale_slot_released = True
            await cam._idle_watchdog_tick()
            client.set_printer_video_stream.assert_not_called()

        _run(run())

    def test_disable_on_home_assistant_stop(self) -> None:
        """A shutdown mid-snapshot does not leave the stream enabled."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client)
            cam._stream_enabled = True
            cam._schedule_disable()
            await cam._async_disable_on_stop(MagicMock())
            client.set_printer_video_stream.assert_called_once_with(enable=False)
            assert cam._stream_enabled is False
            assert cam._pending_disable_task is None

        _run(run())

    def test_failed_disable_keeps_flag_for_watchdog_retry(self) -> None:
        """If the disable fails, the stream is not assumed to be off."""

        async def run() -> None:
            client = _make_client()
            client.set_printer_video_stream = AsyncMock(side_effect=OSError("boom"))
            cam = _cc2_camera(client)
            cam._stream_enabled = True
            await cam._disable_stream()
            assert cam._stream_enabled is True

            # Watchdog retries and succeeds.
            client.set_printer_video_stream = AsyncMock()
            cam._stale_slot_released = True
            await cam._idle_watchdog_tick()
            assert cam._stream_enabled is False

        _run(run())

    def test_cleanup_cancels_debounce_and_disables(self) -> None:
        """Entity removal disables immediately rather than waiting."""

        async def run() -> None:
            client = _make_client()
            cam = _cc2_camera(client)
            cam._stream_enabled = True
            cam._schedule_disable()
            pending = cam._pending_disable_task
            await cam._cleanup_video_lifecycle()
            assert pending.cancelled()
            assert cam._pending_disable_task is None
            client.set_printer_video_stream.assert_called_once_with(enable=False)
            assert cam._stream_enabled is False

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
                patch.object(cam, "_get_frame_hub") as get_hub,
            ):
                image = await cam.async_camera_image()
            assert image == JPEG
            base.assert_awaited_once()
            # No CC2 frame hub on other transports.
            get_hub.assert_not_called()
            assert cam._frame_hub is None

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
