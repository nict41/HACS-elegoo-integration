"""
Tests for the CC2 camera diagnostics and passive mode.

These exist to answer one question from a log alone: does this
integration cause the printer's camera to stop answering after a while?
They cover the printer-reported telemetry (video slot counts and
camera_status), the activity ledger, and the passive mode that sends no
video control commands at all.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import custom_components.elegoo_printer.camera as camera_module
from custom_components.elegoo_printer.camera import ElegooMjpegCamera
from custom_components.elegoo_printer.cc2.client import ElegooCC2Client
from custom_components.elegoo_printer.sdcp.models.enums import ElegooVideoStatus

STREAM_URL = "http://printer.invalid:8080/?action=stream"


def _run(coro):
    """Run an async coroutine to completion (fresh loop)."""
    asyncio.run(coro)


def _attrs(num: int, max_allowed: int = 1, camera_status: int = 1) -> MagicMock:
    attrs = MagicMock()
    attrs.num_video_stream_connected = num
    attrs.max_video_stream_allowed = max_allowed
    attrs.camera_status = camera_status
    return attrs


def _client_with_logger() -> tuple[ElegooCC2Client, MagicMock]:
    """Build a bare CC2 client with telemetry state and a mock logger."""
    client = object.__new__(ElegooCC2Client)
    client.logger = MagicMock()
    client._last_video_slot_count = None
    client._last_camera_status = None
    return client, client.logger


def _camera(*, passive: bool = False) -> ElegooMjpegCamera:
    """Build an ElegooMjpegCamera without running full entity init."""
    client = MagicMock()
    client.is_connected = True
    client.printer_data = MagicMock()
    client.printer_data.attributes = _attrs(0)
    video = MagicMock()
    video.status = ElegooVideoStatus.SUCCESS
    video.video_url = STREAM_URL
    client.printer_data.video = video
    client.get_printer_video = AsyncMock(return_value=video)
    client.set_printer_video_stream = AsyncMock()

    cam = object.__new__(ElegooMjpegCamera)
    cam.hass = MagicMock()
    cam.entity_id = "camera.chamber"
    cam._mjpeg_url = STREAM_URL
    cam._init_video_lifecycle(client)
    cam._is_cc2 = True
    cam._cc2_passive = passive
    return cam


class TestSlotTelemetry:
    """The printer's own video slot count, logged on every change."""

    def test_baseline_logged_once(self) -> None:
        """The first reading establishes a baseline at INFO."""
        client, logger = _client_with_logger()
        client._log_camera_telemetry(_attrs(0))
        assert "baseline" in logger.info.call_args[0][0]

    def test_slot_change_is_logged(self) -> None:
        """A change in the slot count is reported with both values."""
        client, logger = _client_with_logger()
        client._log_camera_telemetry(_attrs(0, max_allowed=4))
        logger.info.reset_mock()
        client._log_camera_telemetry(_attrs(1, max_allowed=4))
        args = logger.info.call_args[0]
        assert args[1] == 0
        assert args[2] == 1

    def test_unchanged_count_is_not_logged(self) -> None:
        """A steady count produces no repeated noise."""
        client, logger = _client_with_logger()
        client._log_camera_telemetry(_attrs(1, max_allowed=4))
        logger.info.reset_mock()
        client._log_camera_telemetry(_attrs(1, max_allowed=4))
        logger.info.assert_not_called()

    def test_exhaustion_is_a_warning(self) -> None:
        """Reaching the maximum is a WARNING, not an INFO."""
        client, logger = _client_with_logger()
        client._log_camera_telemetry(_attrs(0, max_allowed=1))
        client._log_camera_telemetry(_attrs(1, max_allowed=1))
        assert "EXHAUSTED" in logger.warning.call_args[0][0]

    def test_camera_status_drop_is_a_warning(self) -> None:
        """camera_status 1 -> 0 means the printer lost the camera."""
        client, logger = _client_with_logger()
        client._log_camera_telemetry(_attrs(0, camera_status=1))
        logger.warning.reset_mock()
        client._log_camera_telemetry(_attrs(0, camera_status=0))
        args = logger.warning.call_args[0]
        assert "camera_status changed" in args[0]
        assert args[1] == 1
        assert args[2] == 0

    def test_steady_camera_status_is_not_logged(self) -> None:
        """An unchanged camera_status is silent."""
        client, logger = _client_with_logger()
        client._log_camera_telemetry(_attrs(0, camera_status=1))
        logger.warning.reset_mock()
        client._log_camera_telemetry(_attrs(0, camera_status=1))
        logger.warning.assert_not_called()


class TestActivityLedger:
    """The periodic summary of what the camera entity actually did."""

    def test_counts_enables_and_disables(self) -> None:
        """Every video command the entity sends is counted."""

        async def run() -> None:
            cam = _camera()
            await cam._ensure_stream_enabled()
            await cam._disable_stream()
            assert cam._stats["enables_sent"] == 1
            assert cam._stats["disables_sent"] == 1
            assert cam._stats["disable_failures"] == 0

        _run(run())

    def test_counts_disable_failures(self) -> None:
        """A disable that fails is counted separately."""

        async def run() -> None:
            cam = _camera()
            cam._stream_enabled = True
            cam._printer_client.set_printer_video_stream = AsyncMock(
                side_effect=OSError("boom")
            )
            await cam._disable_stream()
            assert cam._stats["disable_failures"] == 1

        _run(run())

    def test_summary_is_silent_without_activity(self) -> None:
        """No activity, no summary."""

        async def run() -> None:
            cam = _camera()
            with patch.object(camera_module.LOGGER, "info") as info:
                cam._log_activity_summary()
            info.assert_not_called()

        _run(run())

    def test_summary_reports_counters_and_printer_state(self) -> None:
        """The summary lines up our actions against the printer's view."""

        async def run() -> None:
            cam = _camera()
            cam._printer_client.printer_data.attributes = _attrs(
                3, max_allowed=4, camera_status=0
            )
            cam._stats["image_requests"] = 7
            with patch.object(camera_module.LOGGER, "info") as info:
                cam._log_activity_summary()
            args = info.call_args[0]
            assert "image_requests=7" in args[2]
            assert args[3] == 3
            assert args[4] == 4
            assert args[5] == 0

        _run(run())

    def test_summary_is_rate_limited(self) -> None:
        """The summary does not repeat on every watchdog tick."""

        async def run() -> None:
            cam = _camera()
            cam._stats["image_requests"] = 1
            with patch.object(camera_module.LOGGER, "info") as info:
                cam._log_activity_summary()
                cam._log_activity_summary()
            assert info.call_count == 1

        _run(run())


class TestPassiveMode:
    """Passive mode sends no video control commands at all."""

    def test_no_enable_command_is_sent(self) -> None:
        """The stream URL is used without touching the printer."""

        async def run() -> None:
            cam = _camera(passive=True)
            await cam._update_stream_url()
            cam._printer_client.get_printer_video.assert_not_called()
            cam._printer_client.set_printer_video_stream.assert_not_called()
            assert cam._mjpeg_url == STREAM_URL

        _run(run())

    def test_no_disable_command_is_sent(self) -> None:
        """There is nothing enabled, so nothing is disabled."""

        async def run() -> None:
            cam = _camera(passive=True)
            cam._stream_enabled = True
            await cam._disable_stream()
            cam._printer_client.set_printer_video_stream.assert_not_called()
            assert cam._stream_enabled is False

        _run(run())

    def test_snapshot_sends_nothing_to_the_printer(self) -> None:
        """A whole snapshot cycle issues zero video commands."""

        async def run() -> None:
            cam = _camera(passive=True)
            with patch.object(
                cam, "_grab_frame", AsyncMock(return_value=b"\xff\xd8\xff\xd9")
            ):
                image = await cam.async_camera_image()
            assert image is not None
            cam._printer_client.get_printer_video.assert_not_called()
            cam._printer_client.set_printer_video_stream.assert_not_called()
            cam._cancel_pending_disable()

        _run(run())

    def test_active_mode_still_sends_commands(self) -> None:
        """The default remains the normal enable/disable lifecycle."""

        async def run() -> None:
            cam = _camera(passive=False)
            cam._mjpeg_url = None
            await cam._update_stream_url()
            cam._printer_client.get_printer_video.assert_called_once_with(enable=True)

        _run(run())


class TestPortProbeRearm:
    """The port probe can run again once the camera has degraded."""

    def test_probe_rearms_after_interval(self) -> None:
        """A probe while healthy and another once broken are both allowed."""

        async def run() -> None:
            cam = _camera()

            async def fake_open(_host: str, _port: int):
                raise ConnectionRefusedError

            with (
                patch.object(camera_module.asyncio, "open_connection", fake_open),
                patch.object(camera_module.LOGGER, "warning") as warn,
            ):
                await cam._probe_camera_ports(STREAM_URL)
                await cam._probe_camera_ports(STREAM_URL)
                assert warn.call_count == 1
                # Pretend the interval has elapsed.
                cam._last_port_probe -= camera_module.PORT_PROBE_INTERVAL + 1
                await cam._probe_camera_ports(STREAM_URL)
                assert warn.call_count == 2

        _run(run())
