"""
Tests for the CC2 camera diagnostics and passive mode.

These exist to answer one question from a log alone: does this
integration cause the printer's camera to stop answering after a while?
They cover the printer-reported telemetry (video slot counts and
camera_status), the activity ledger, and the passive mode that sends no
video control commands at all.
"""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import custom_components.elegoo_printer.camera as camera_module
from custom_components.elegoo_printer.camera import ElegooMjpegCamera
from custom_components.elegoo_printer.cc2.client import ElegooCC2Client
from custom_components.elegoo_printer.const import CC2_VIDEO_PATH, CC2_VIDEO_PORT
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


async def _tick_port_watchdog(cam) -> None:
    """Let the port watchdog complete at least one poll, then stop it."""
    task = asyncio.create_task(cam._camera_port_watchdog())
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


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


class TestPerGrabSummary:
    """One INFO line per snapshot, readable without debug logging."""

    def test_successful_grab_logs_result_and_size(self) -> None:
        """A good grab reports ok, its size and a sequence number."""

        async def run() -> None:
            cam = _camera(passive=True)
            frame = b"\xff\xd8payload\xff\xd9"
            with (
                patch.object(cam, "_grab_frame", AsyncMock(return_value=frame)),
                patch.object(camera_module.LOGGER, "info") as info,
            ):
                await cam.async_camera_image()
            line = next(c for c in info.call_args_list if "grab #%d" in c[0][0])
            assert line[0][1] == 1  # sequence
            assert line[0][3] == "ok"
            assert line[0][4] == len(frame)
            cam._cancel_pending_disable()

        _run(run())

    def test_failed_grab_logs_failed(self) -> None:
        """A grab that returns nothing is recorded as failed, not dropped."""

        async def run() -> None:
            cam = _camera(passive=True)
            with (
                patch.object(cam, "_grab_frame", AsyncMock(return_value=None)),
                patch.object(
                    camera_module.MjpegCamera,
                    "async_camera_image",
                    AsyncMock(return_value=None),
                ),
                patch.object(camera_module.LOGGER, "info") as info,
            ):
                await cam.async_camera_image()
            line = next(c for c in info.call_args_list if "grab #%d" in c[0][0])
            assert line[0][3] == "failed"
            cam._cancel_pending_disable()

        _run(run())

    def test_sequence_increments_across_grabs(self) -> None:
        """Sequence numbers let interleaved lines be put back in order."""

        async def run() -> None:
            cam = _camera(passive=True)
            with (
                patch.object(
                    cam, "_grab_frame", AsyncMock(return_value=b"\xff\xd8\xff\xd9")
                ),
                patch.object(camera_module.LOGGER, "info") as info,
            ):
                await cam.async_camera_image()
                await cam.async_camera_image()
            seqs = [c[0][1] for c in info.call_args_list if "grab #%d" in c[0][0]]
            assert seqs == [1, 2]
            cam._cancel_pending_disable()

        _run(run())

    def test_heartbeat_logs_even_with_no_activity(self) -> None:
        """The summary is a heartbeat, so idle periods are still visible."""

        async def run() -> None:
            cam = _camera()
            with patch.object(camera_module.LOGGER, "info") as info:
                cam._log_activity_summary()
            assert any("activity for" in c[0][0] for c in info.call_args_list)

        _run(run())


class TestVideoResponseUrl:
    """The method-1042 reply carries the stream URL under "url"."""

    def _client(self) -> tuple[ElegooCC2Client, MagicMock]:
        client = object.__new__(ElegooCC2Client)
        client.logger = MagicMock()
        client.printer_ip = "192.168.8.128"
        client.printer_data = MagicMock()
        return client, client.logger

    def test_url_key_is_used(self) -> None:
        """Firmware 02.01.00.00 sends "url", which must be honoured."""
        client, _ = self._client()
        client._handle_video_response(
            {"error_code": 0, "url": "http://192.168.8.128:8080/?action=stream"}
        )
        assert (
            client.printer_data.video.video_url
            == "http://192.168.8.128:8080/?action=stream"
        )

    def test_video_url_key_still_works(self) -> None:
        """Other firmware may use video_url; keep accepting it."""
        client, _ = self._client()
        client._handle_video_response(
            {"error_code": 0, "video_url": "http://host:1234/s"}
        )
        assert client.printer_data.video.video_url == "http://host:1234/s"

    def test_missing_url_warns_and_falls_back(self) -> None:
        """A reply with neither key is a warning, not a silent guess."""
        client, logger = self._client()
        client._handle_video_response({"error_code": 0})
        assert "carried no url/video_url" in logger.warning.call_args[0][0]
        assert client.printer_data.video.video_url == (
            f"http://192.168.8.128:{CC2_VIDEO_PORT}{CC2_VIDEO_PATH}"
        )


class TestCameraPortWatchdog:
    """Transitions of the camera's TCP port, logged as they happen."""

    def test_startup_state_logged_once(self) -> None:
        """The first observation is a baseline at INFO."""

        async def run() -> None:
            cam = _camera()
            with (
                patch.object(cam, "_is_camera_port_open", AsyncMock(return_value=True)),
                patch.object(camera_module, "CAMERA_PORT_POLL_INTERVAL", 0),
                patch.object(camera_module.LOGGER, "info") as info,
            ):
                await _tick_port_watchdog(cam)
            assert any("at startup" in c[0][0] for c in info.call_args_list)

        _run(run())

    def test_open_to_closed_is_a_warning(self) -> None:
        """The camera dying mid-session is a WARNING with the command count."""

        async def run() -> None:
            cam = _camera()
            cam._camera_port_open = True
            cam._printer_client._video_command_count = 42
            with (
                patch.object(
                    cam, "_is_camera_port_open", AsyncMock(return_value=False)
                ),
                patch.object(camera_module, "CAMERA_PORT_POLL_INTERVAL", 0),
                patch.object(camera_module.LOGGER, "warning") as warn,
            ):
                await _tick_port_watchdog(cam)
            args = warn.call_args[0]
            assert args[2] == "OPEN"
            assert args[3] == "CLOSED"
            assert args[5] == 42

        _run(run())

    def test_steady_state_is_not_logged(self) -> None:
        """No transition, no line."""

        async def run() -> None:
            cam = _camera()
            cam._camera_port_open = True
            with (
                patch.object(cam, "_is_camera_port_open", AsyncMock(return_value=True)),
                patch.object(camera_module, "CAMERA_PORT_POLL_INTERVAL", 0),
                patch.object(camera_module.LOGGER, "warning") as warn,
                patch.object(camera_module.LOGGER, "info") as info,
            ):
                await _tick_port_watchdog(cam)
            warn.assert_not_called()
            assert not any("camera port" in c[0][0] for c in info.call_args_list)

        _run(run())


class TestPassiveSendsNothingAtStartup:
    """Passive mode's promise: zero video commands, including on load."""

    def test_no_startup_disable_in_passive_mode(self) -> None:
        """The stale-slot release is skipped so the A/B stays clean."""

        async def run() -> None:
            cam = _camera(passive=True)
            await cam._idle_watchdog_tick()
            cam._printer_client.set_printer_video_stream.assert_not_called()

        _run(run())

    def test_active_mode_still_releases_at_startup(self) -> None:
        """Non-passive keeps the recovery behaviour."""

        async def run() -> None:
            cam = _camera(passive=False)
            await cam._idle_watchdog_tick()
            cam._printer_client.set_printer_video_stream.assert_called_once_with(
                enable=False
            )

        _run(run())


class TestEnableCounting:
    """enables_sent must reflect the CC2 grab path, not just the mixin."""

    def test_update_stream_url_counts_the_enable(self) -> None:
        """The path Run A actually used was previously uncounted."""

        async def run() -> None:
            cam = _camera(passive=False)
            cam._mjpeg_url = None
            await cam._update_stream_url()
            assert cam._stats["enables_sent"] == 1

        _run(run())
