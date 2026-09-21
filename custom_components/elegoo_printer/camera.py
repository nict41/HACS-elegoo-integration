"""Camera platform for Elegoo printer."""

import asyncio
import contextlib
from http import HTTPStatus
from typing import TYPE_CHECKING

import aiohttp
from aiohttp import web
from haffmpeg.camera import CameraMjpeg
from homeassistant.components.camera import Camera, CameraEntityFeature
from homeassistant.components.ffmpeg import (
    DOMAIN,
    async_get_image,
)
from homeassistant.components.mjpeg.camera import MjpegCamera
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import (
    async_aiohttp_proxy_stream,
    async_get_clientsession,
)
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from propcache.api import cached_property
from yarl import URL

from custom_components.elegoo_printer.const import (
    CONF_CAMERA_ENABLED,
    LOGGER,
    VIDEO_ENDPOINT,
    VIDEO_PORT,
)
from custom_components.elegoo_printer.data import ElegooPrinterConfigEntry
from custom_components.elegoo_printer.definitions import (
    PRINTER_FFMPEG_CAMERAS,
    PRINTER_MJPEG_CAMERAS,
    ElegooPrinterSensorEntityDescription,
)
from custom_components.elegoo_printer.entity import ElegooPrinterEntity
from custom_components.elegoo_printer.sdcp.models.enums import (
    ElegooVideoStatus,
    PrinterType,
    TransportType,
)
from custom_components.elegoo_printer.sdcp.models.printer import PrinterData

from .coordinator import ElegooDataUpdateCoordinator

if TYPE_CHECKING:
    from custom_components.elegoo_printer.websocket.client import ElegooPrinterClient

# Graceful ffmpeg shutdown timeouts
FFMPEG_QUIT_TIMEOUT = 10  # seconds to wait after sending 'q' to ffmpeg
FFMPEG_TERMINATE_TIMEOUT = 5  # seconds to wait after SIGTERM before SIGKILL
NATIVE_STREAM_IDLE_TIMEOUT = 600  # 10 minutes — clear native stream flag after idle
IDLE_WATCHDOG_INTERVAL = 60  # seconds between idle checks

# CC2 chamber camera snapshot tuning. The printer acknowledges the enable
# command (method 1042) before its MJPEG server is necessarily accepting
# connections, so a grab issued immediately after the ack can hit a socket
# that is not listening yet.
STREAM_READY_TIMEOUT = 5.0  # max seconds to wait for the stream server
STREAM_READY_INITIAL_DELAY = 0.2  # first backoff step
STREAM_READY_MAX_DELAY = 1.0  # backoff ceiling
FRAME_FETCH_TIMEOUT = 10.0  # seconds for a single-frame grab
FRAME_FETCH_MAX_BYTES = 4 * 1024 * 1024  # give up rather than read a stream forever
BUFFER_SIZE = 102400  # matches homeassistant.components.mjpeg.camera
DISABLE_DEBOUNCE_DELAY = 5.0  # keep video on briefly so back-to-back grabs reuse it
GRAB_FAILURE_LOG_INTERVAL = 300.0  # seconds between repeated grab-failure warnings

# Candidate snapshot/stream paths probed once, at WARNING, after a CC2 grab
# fails. NONE of these are documented for CC2 firmware — the probe exists to
# find out which (if any) the camera actually serves, without guessing in the
# hot path.
CC2_PROBE_PATHS = (
    "/?action=snapshot",
    "/?action=stream",
    "/snapshot",
    "/stream",
    "/video",
    "/",
)


class ElegooCameraMjpeg(CameraMjpeg):
    """
    CameraMjpeg with graceful shutdown: quit -> SIGTERM -> SIGKILL.

    ffmpeg's RTSP demuxer sends RTSP TEARDOWN on SIGTERM, which tells the
    printer to decrement its session counter. SIGKILL bypasses this entirely.
    """

    async def close(self, shutdown_timeout: int = FFMPEG_QUIT_TIMEOUT) -> None:
        """
        Stop ffmpeg with graceful shutdown sequence.

        Arguments:
            shutdown_timeout: Seconds to wait after sending 'q' before SIGTERM.

        """
        if not self.is_running:
            self._clear()
            return

        # Step 1: Send 'q' to ffmpeg stdin (ffmpeg's interactive quit)
        quit_timed_out = False
        try:
            self._proc.stdin.write(b"q")
            async with asyncio.timeout(shutdown_timeout):
                await self._proc.wait()
        except (BrokenPipeError, RuntimeError, OSError):
            # stdin is closed or process already died — skip to SIGTERM
            LOGGER.debug("FFmpeg stdin unavailable, skipping to SIGTERM")
        except asyncio.TimeoutError:
            quit_timed_out = True
        else:
            LOGGER.debug("Closed FFmpeg process gracefully (quit)")
            self._clear()
            return

        if not quit_timed_out and not self.is_running:
            # Process may have already exited after stdin error
            self._clear()
            return

        # Step 2: SIGTERM — ffmpeg sends RTSP TEARDOWN on SIGTERM
        try:
            self._proc.terminate()  # SIGTERM
            async with asyncio.timeout(FFMPEG_TERMINATE_TIMEOUT):
                await self._proc.wait()
            LOGGER.debug("Closed FFmpeg process (SIGTERM)")
        except ProcessLookupError:
            # Process already exited — treat as success
            LOGGER.debug("FFmpeg process already exited during SIGTERM")
        except asyncio.TimeoutError:
            # Step 3: SIGKILL as absolute last resort
            LOGGER.warning("SIGTERM timed out, escalating to SIGKILL")
            self.kill()  # reuse base class SIGKILL + background communicate task

        self._clear()


class ElegooVideoStreamLifecycle(ElegooPrinterEntity):
    """
    Ref-counted lifecycle for the printer's video stream.

    Printers advertise a fixed number of concurrent video stream
    connections (num_video_stream_connected vs. max_video_stream_allowed)
    and once the stream is enabled it stays active on the printer side
    until explicitly disabled. Leaving it enabled with no viewers occupies
    a slot (which can block other consumers and requires a printer reboot
    to release), so this mixin:

    - Enables the video when the first viewer (MJPEG stream, transient
      image grab, or native stream) appears
    - Disables it when the last viewer disconnects
    - An idle watchdog re-attempts failed disables and clears stale
      native stream flags
    - Disables on entity removal to clean up residual state

    The mixin does not own entity state. Camera classes call
    ``_init_video_lifecycle(client)`` inside their own ``__init__`` once
    ``self._printer_client`` is available.
    """

    def _init_video_lifecycle(self, client: "ElegooPrinterClient") -> None:
        """Initialize stream lifecycle state on this camera entity."""
        self._printer_client = client
        self._active_mjpeg_streams = 0
        self._transient_viewers = 0
        self._native_stream_active = False
        self._stream_enabled = False
        self._last_activity = 0.0
        self._idle_watchdog_task = None
        # CC2-only state. Defaults to off so SDCP/resin printers keep the
        # original behaviour untouched; ElegooMjpegCamera.__init__ turns it on
        # for TransportType.CC2_MQTT.
        self._is_cc2 = False
        self._stream_lock = asyncio.Lock()
        self._pending_disable_task: asyncio.Task | None = None
        self._last_grab_failure_log = 0.0
        self._endpoints_probed = False

    def _log_grab_failure(self, reason: str, *args: object) -> None:
        """
        Log why an image grab returned None, rate-limited.

        The first failure (and one every GRAB_FAILURE_LOG_INTERVAL after
        that) is a WARNING so it shows up without debug logging enabled;
        the rest are DEBUG so a snapshot automation cannot flood the log.
        """
        now = asyncio.get_running_loop().time()
        if now - self._last_grab_failure_log >= GRAB_FAILURE_LOG_INTERVAL:
            self._last_grab_failure_log = now
            LOGGER.warning(
                "Camera image grab failed for %s: " + reason,
                self.entity_id,
                *args,
            )
        else:
            LOGGER.debug(
                "Camera image grab failed for %s: " + reason,
                self.entity_id,
                *args,
            )

    def _capacity_counters(self) -> tuple[int, int]:
        """Return (num_video_stream_connected, max_video_stream_allowed)."""
        attrs = self._printer_client.printer_data.attributes
        return (
            getattr(attrs, "num_video_stream_connected", 0) or 0,
            getattr(attrs, "max_video_stream_allowed", 0) or 0,
        )

    def _cancel_pending_disable(self) -> None:
        """Cancel a debounced disable so the enabled stream is reused."""
        task = self._pending_disable_task
        if task is not None and not task.done():
            task.cancel()
        self._pending_disable_task = None

    def _schedule_disable(self) -> None:
        """
        Disable the video stream after DISABLE_DEBOUNCE_DELAY seconds.

        Back-to-back snapshots land inside the delay and reuse the already
        enabled stream instead of toggling the printer off and on again.

        The debounce is best effort: if the task is cancelled or lost, the
        idle watchdog still disables an enabled stream with no viewers on
        its next pass, so a slot is never leaked indefinitely.
        """
        self._cancel_pending_disable()
        self._pending_disable_task = asyncio.create_task(self._debounced_disable())

    async def _debounced_disable(self) -> None:
        """Wait out the debounce window, then disable if still idle."""
        await asyncio.sleep(DISABLE_DEBOUNCE_DELAY)
        # Take the lock so a disable can never interleave with a grab that
        # started in the meantime.
        async with self._stream_lock:
            if not self._has_active_viewers():
                await self._disable_stream()

    def _is_over_capacity(self) -> bool:
        """Check if the printer is over capacity."""
        num_connected, max_allowed = self._capacity_counters()
        return num_connected >= max_allowed

    def _has_active_viewers(self) -> bool:
        """Check if any viewer type is currently active."""
        return (
            self._active_mjpeg_streams > 0
            or self._transient_viewers > 0
            or self._native_stream_active
        )

    async def _ensure_stream_enabled(self) -> None:
        """
        Enable printer video if not already enabled.

        Idempotent — safe to call when already enabled.
        On failure, _stream_enabled is NOT set (may retry later).
        """
        if self._stream_enabled:
            return
        if not self._printer_client.is_connected:
            LOGGER.debug(
                "Printer client not connected, deferring video enable for %s",
                self.entity_id,
            )
            return
        try:
            video = await self._printer_client.get_printer_video(enable=True)
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(
                "Exception enabling printer video for %s: %s",
                self.entity_id,
                e,
            )
            return
        if video.status == ElegooVideoStatus.SUCCESS:
            self._stream_enabled = True
            LOGGER.debug("Enabled printer video for %s", self.entity_id)
        else:
            LOGGER.warning(
                "Failed to enable printer video for %s: %s",
                self.entity_id,
                video.status,
            )

    async def _disable_stream(self) -> None:
        """
        Disable printer video.

        On failure, _stream_enabled stays True (video may still be on
        printer). The idle watchdog will re-attempt on subsequent
        intervals.
        """
        if not self._stream_enabled:
            return
        try:
            await self._printer_client.set_printer_video_stream(enable=False)
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(
                "Failed to disable printer video for %s (may be over capacity): %s",
                self.entity_id,
                e,
            )
            # Don't clear flag — video may still be enabled on printer
            return
        self._stream_enabled = False
        LOGGER.debug("Disabled printer video for %s", self.entity_id)

    async def _get_stream_url(self) -> str | None:
        """
        Get the stream URL from cached printer data.

        Does NOT toggle the printer video — reads the URL cached by the
        last call to get_printer_video(). Callers must ensure the video
        is enabled via _ensure_stream_enabled() before calling this method.
        """
        if (not self._printer_client.is_connected) or self._is_over_capacity():
            return None
        video_url = self._printer_client.printer_data.video.video_url
        if video_url:
            LOGGER.debug(
                "stream_source: Using cached stream URL: %s",
                video_url,
            )
            return video_url
        return None

    async def _idle_watchdog_tick(self) -> None:
        """
        Run a single watchdog pass.

        1. If no viewers are active and video is enabled, attempt to
           disable it (handles failed disables from the normal
           disconnect path).
        2. If the native stream has been idle for
           NATIVE_STREAM_IDLE_TIMEOUT, clear the native-stream flag
           (allows a future disable attempt).
        """
        if self._stream_enabled and not self._has_active_viewers():
            await self._disable_stream()
        if (
            self._native_stream_active
            and self._last_activity > 0
            and asyncio.get_running_loop().time() - self._last_activity
            > NATIVE_STREAM_IDLE_TIMEOUT
        ):
            LOGGER.debug(
                "Native stream idle for %.0fs, clearing flag for %s",
                NATIVE_STREAM_IDLE_TIMEOUT,
                self.entity_id,
            )
            self._native_stream_active = False

    async def _idle_watchdog(self) -> None:
        """
        Periodically check for idle conditions and clean up.

        Runs every IDLE_WATCHDOG_INTERVAL seconds. See the tick for the
        two responsibilities (disabled-when-idle, stale native flag).
        """
        while True:
            try:
                await asyncio.sleep(IDLE_WATCHDOG_INTERVAL)
                await self._idle_watchdog_tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                LOGGER.exception("Idle watchdog error for %s", self.entity_id)

    async def _cleanup_video_lifecycle(self) -> None:
        """Cancel the idle watchdog and release the video stream state."""
        # Drop any debounced disable — the explicit _disable_stream() below
        # supersedes it, and a pending task would outlive the entity.
        self._cancel_pending_disable()
        if self._idle_watchdog_task is not None:
            self._idle_watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._idle_watchdog_task
            self._idle_watchdog_task = None
        self._active_mjpeg_streams = 0
        self._transient_viewers = 0
        self._native_stream_active = False
        await self._disable_stream()

    async def async_added_to_hass(self) -> None:
        """Start the idle watchdog when the entity is added."""
        await super().async_added_to_hass()
        self._idle_watchdog_task = asyncio.create_task(self._idle_watchdog())

    async def async_will_remove_from_hass(self) -> None:
        """
        Clean up when the entity is removed from Home Assistant.

        Cancels the idle watchdog, resets stream state and disables the
        printer video.
        """
        await super().async_will_remove_from_hass()
        await self._cleanup_video_lifecycle()


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ElegooPrinterConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Asynchronously sets up Elegoo camera entities."""
    coordinator: ElegooDataUpdateCoordinator = config_entry.runtime_data.coordinator
    printer_type = coordinator.config_entry.runtime_data.api.printer.printer_type

    if printer_type == PrinterType.FDM:
        LOGGER.debug(f"Adding {len(PRINTER_MJPEG_CAMERAS)} Camera entities")
        for camera in PRINTER_MJPEG_CAMERAS:
            async_add_entities(
                [ElegooMjpegCamera(hass, coordinator, camera)], update_before_add=True
            )
    elif printer_type == PrinterType.RESIN:
        LOGGER.debug(f"Adding {len(PRINTER_FFMPEG_CAMERAS)} Camera entities")
        for camera in PRINTER_FFMPEG_CAMERAS:
            async_add_entities(
                [ElegooStreamCamera(hass, coordinator, camera)],
                update_before_add=True,
            )


class ElegooStreamCamera(ElegooVideoStreamLifecycle, Camera):
    """Representation of a camera that streams from an Elegoo printer."""

    def __init__(
        self,
        hass: HomeAssistant,  # noqa: ARG002
        coordinator: ElegooDataUpdateCoordinator,
        description: ElegooPrinterSensorEntityDescription,
    ) -> None:
        """Initialize an Elegoo stream camera entity."""
        Camera.__init__(self)
        ElegooPrinterEntity.__init__(self, coordinator)

        self.entity_description = description
        self._printer_client: ElegooPrinterClient = (
            coordinator.config_entry.runtime_data.api.client
        )
        self._attr_name = description.name
        self._attr_unique_id = coordinator.generate_unique_id(description.key)
        self._attr_entity_registry_enabled_default = coordinator.config_entry.data.get(
            CONF_CAMERA_ENABLED, False
        )

        # For MJPEG stream
        self._extra_ffmpeg_arguments = (
            "-rtsp_transport udp -fflags nobuffer -err_detect ignore_err"
        )
        self._active_mjpeg_processes: set[ElegooCameraMjpeg] = set()
        self._init_video_lifecycle(self._printer_client)

    @cached_property
    def supported_features(self) -> CameraEntityFeature:
        """Return supported features."""
        return self._attr_supported_features

    async def handle_async_mjpeg_stream(
        self, request: web.Request
    ) -> web.StreamResponse:
        """
        Generate an HTTP MJPEG stream from the camera.

        Ref-counted: enables video on first viewer, disables on last.
        Uses ElegooCameraMjpeg for graceful SIGTERM shutdown.
        """
        mjpeg_stream: ElegooCameraMjpeg | None = None

        # Enable stream if first viewer
        if not self._has_active_viewers():
            await self._ensure_stream_enabled()

        try:
            stream_url = await self._get_stream_url()
            if not stream_url:
                return web.Response(
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                    reason="Stream URL not available",
                )

            ffmpeg_manager = self.hass.data[DOMAIN]
            mjpeg_stream = ElegooCameraMjpeg(ffmpeg_manager.binary)
            await mjpeg_stream.open_camera(
                stream_url, extra_cmd=self._extra_ffmpeg_arguments
            )

            self._active_mjpeg_streams += 1
            self._active_mjpeg_processes.add(mjpeg_stream)
            self._last_activity = asyncio.get_running_loop().time()

            stream_reader = await mjpeg_stream.get_reader()
            return await async_aiohttp_proxy_stream(
                self.hass,
                request,
                stream_reader,
                ffmpeg_manager.ffmpeg_stream_content_type,
            )
        finally:
            if mjpeg_stream is not None:
                self._active_mjpeg_streams = max(0, self._active_mjpeg_streams - 1)
                self._active_mjpeg_processes.discard(mjpeg_stream)
                await mjpeg_stream.close(shutdown_timeout=FFMPEG_QUIT_TIMEOUT)
            # Disable stream if last viewer
            if not self._has_active_viewers():
                await self._disable_stream()

    async def stream_source(self) -> str | None:
        """
        Return the source of the stream.

        Enables video for native HA streaming. Uses idle watchdog to
        disable after NATIVE_STREAM_IDLE_TIMEOUT of no activity.
        """
        if not self._native_stream_active:
            await self._ensure_stream_enabled()
            # Only set flag if video was actually enabled
            if not self._stream_enabled:
                return None
            self._native_stream_active = True

        stream_url = await self._get_stream_url()
        if not stream_url:
            return None

        self._last_activity = asyncio.get_running_loop().time()
        return stream_url

    async def async_camera_image(
        self,
        width: int | None = None,  # noqa: ARG002
        height: int | None = None,  # noqa: ARG002
    ) -> bytes | None:
        """
        Return a still image from the camera.

        Treats the image grab as a transient viewer — enables video if
        needed, but only disables if no other viewers are active.

        Note: This path uses HA's async_get_image() which spawns its own
        ffmpeg process. That process does NOT get graceful SIGTERM shutdown,
        so individual image grabs may leak RTSP sessions. The _transient_viewers
        counter prevents this path from disabling an active MJPEG stream.
        """
        # Enable stream if no other viewers are active (check before increment)
        if not self._has_active_viewers():
            await self._ensure_stream_enabled()
        self._transient_viewers += 1

        try:
            stream_url = await self._get_stream_url()
            if not stream_url:
                return None
            return await async_get_image(
                self.hass,
                input_source=stream_url,
            )
        except Exception as e:  # noqa: BLE001
            LOGGER.error(
                "Failed to get camera image via ffmpeg (ffmpeg may be missing): %s", e
            )
            return None
        finally:
            self._transient_viewers = max(0, self._transient_viewers - 1)
            # Only disable if no other viewers are active
            if not self._has_active_viewers():
                await self._disable_stream()

    async def async_will_remove_from_hass(self) -> None:
        """
        Clean up when the entity is removed from Home Assistant.

        Closes any in-flight MJPEG processes (camera-specific state),
        then delegates to the lifecycle for watchdog cancellation and
        stream disabling.
        """
        for proc in self._active_mjpeg_processes.copy():
            await proc.close(shutdown_timeout=FFMPEG_QUIT_TIMEOUT)
        self._active_mjpeg_processes.clear()
        await super().async_will_remove_from_hass()


class ElegooMjpegCamera(ElegooVideoStreamLifecycle, MjpegCamera):
    """Representation of an MjpegCamera."""

    def __init__(
        self,
        hass: HomeAssistant,  # noqa: ARG002
        coordinator: ElegooDataUpdateCoordinator,
        description: ElegooPrinterSensorEntityDescription,
    ) -> None:
        """
        Initialize an Elegoo MJPEG camera entity.

        Arguments:
            hass: The Home Assistant instance.
            coordinator: The data update coordinator.
            description: The entity description.

        """
        # Use centralized proxy with MainboardID routing
        printer = coordinator.config_entry.runtime_data.api.printer
        if printer.proxy_enabled:
            external_ip = getattr(printer, "external_ip", None)
            proxy_ip = PrinterData.get_local_ip(printer.ip_address, external_ip)
            # Use centralized proxy on port 3031 with MainboardID as query parameter
            mjpeg_url = f"http://{proxy_ip}:{VIDEO_PORT}/video?id={printer.id}"
        else:
            # Direct HTTP MJPEG stream from the printer
            mjpeg_url = f"http://{printer.ip_address}:{VIDEO_PORT}/{VIDEO_ENDPOINT}"

        MjpegCamera.__init__(
            self,
            name=f"{description.name}",
            mjpeg_url=mjpeg_url,
            still_image_url=None,  # This camera does not have a separate still URL
            unique_id=coordinator.generate_unique_id(description.key),
        )

        ElegooPrinterEntity.__init__(self, coordinator)
        self.entity_description = description
        self._printer_client: ElegooPrinterClient = (
            coordinator.config_entry.runtime_data.api.client
        )
        self._init_video_lifecycle(self._printer_client)
        # Only the CC2 takes the readiness-wait/direct-grab path; CC1 and the
        # SDCP printers keep the original behaviour.
        self._is_cc2 = printer.transport_type == TransportType.CC2_MQTT

    @staticmethod
    def _normalize_video_url(video_url: str | None) -> str | None:
        """
        Check if video_url starts with 'http://' and adds it if missing.

        Arguments:
            video_url: The video URL to normalize.

        Returns:
            Normalized video URL string, or None if invalid/empty.

        """
        if not video_url:
            return None

        video_url = video_url.strip()
        if not video_url:
            return None

        if not video_url.startswith("http://"):
            video_url = "http://" + video_url

        return video_url

    async def _update_stream_url(self) -> None:
        """
        Update the MJPEG stream URL and manage video state.

        Ref-counted like the rest of the lifecycle: the update is
        re-requested only when the video is not enabled, or when the
        video is enabled but the URL is mismatched (retries are safe
        because an already-enabled stream tolerates a subsequent enable).
        Over-capacity/disconnected printers are left untouched.
        """
        if self._stream_enabled and self._mjpeg_url:
            # URL still valid from when the stream was enabled
            return
        if (not self._printer_client.is_connected) or self._is_over_capacity():
            self._mjpeg_url = None
            return
        video = await self._printer_client.get_printer_video(enable=True)
        if video.status == ElegooVideoStatus.SUCCESS:
            self._stream_enabled = True
            video_url = self._normalize_video_url(video.video_url)
            self._mjpeg_url = video_url
            if not video_url:
                LOGGER.debug("stream_source: Empty or invalid video URL from printer")
            else:
                LOGGER.debug("stream_source: Using video url: %s", video_url)
        else:
            LOGGER.debug("stream_source: Failed to get video stream: %s", video.status)
            self._stream_enabled = False
            self._mjpeg_url = None

    async def _wait_for_stream_ready(self, url: str) -> bool:
        """
        Wait until the printer's MJPEG server answers on `url`.

        The CC2 acknowledges the enable command before its camera server
        is necessarily listening, so grabbing a frame straight after the
        ack can hit a closed socket. Polls with backoff for at most
        STREAM_READY_TIMEOUT seconds.

        Returns:
            True if the server answered, False if it never did.

        """
        session = async_get_clientsession(self.hass)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + STREAM_READY_TIMEOUT
        delay = STREAM_READY_INITIAL_DELAY
        attempt = 0
        last_error = "no attempt made"

        while loop.time() < deadline:
            attempt += 1
            try:
                # Headers only: enough to prove the server is listening
                # without pulling frames we are about to discard.
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=STREAM_READY_MAX_DELAY * 2)
                ) as response:
                    LOGGER.debug(
                        "Stream readiness probe for %s attempt %d: HTTP %d (%s)",
                        self.entity_id,
                        attempt,
                        response.status,
                        response.headers.get("Content-Type", "no content-type"),
                    )
                    if response.status < HTTPStatus.BAD_REQUEST:
                        return True
                    last_error = f"HTTP {response.status}"
            except (TimeoutError, aiohttp.ClientError, OSError) as err:
                last_error = f"{type(err).__name__}: {err}"
                LOGGER.debug(
                    "Stream readiness probe for %s attempt %d failed: %s",
                    self.entity_id,
                    attempt,
                    last_error,
                )

            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 2, STREAM_READY_MAX_DELAY)

        self._log_grab_failure(
            "stream server at %s not ready after %.1fs (%d attempts, last: %s)",
            url,
            STREAM_READY_TIMEOUT,
            attempt,
            last_error,
        )
        return False

    async def _fetch_single_frame(self, url: str) -> bytes | None:
        """
        Pull one JPEG frame from the MJPEG stream and close the connection.

        Holding the stream open for as short a time as possible matters
        on the CC2, which advertises very few concurrent video
        connections. Unlike Home Assistant's MjpegCamera image path this
        checks the HTTP status and logs it, so a wrong URL or path shows
        up as a real message instead of a silent None.
        """
        session = async_get_clientsession(self.hass)
        try:
            async with asyncio.timeout(FRAME_FETCH_TIMEOUT), session.get(url) as resp:
                content_type = resp.headers.get("Content-Type", "no content-type")
                if resp.status != HTTPStatus.OK:
                    self._log_grab_failure(
                        "stream URL %s returned HTTP %d (%s)",
                        url,
                        resp.status,
                        content_type,
                    )
                    return None

                LOGGER.debug(
                    "Reading frame for %s from %s (HTTP %d, %s)",
                    self.entity_id,
                    url,
                    resp.status,
                    content_type,
                )
                data = b""
                async for chunk in resp.content.iter_chunked(BUFFER_SIZE):
                    data += chunk
                    jpg_end = data.find(b"\xff\xd9")
                    if jpg_end == -1:
                        if len(data) > FRAME_FETCH_MAX_BYTES:
                            break
                        continue
                    jpg_start = data.find(b"\xff\xd8")
                    if jpg_start == -1:
                        if len(data) > FRAME_FETCH_MAX_BYTES:
                            break
                        continue
                    return data[jpg_start : jpg_end + 2]

                self._log_grab_failure(
                    "no JPEG frame in %d bytes from %s (%s)",
                    len(data),
                    url,
                    content_type,
                )
        except TimeoutError:
            self._log_grab_failure(
                "timed out after %.0fs reading a frame from %s",
                FRAME_FETCH_TIMEOUT,
                url,
            )
        except (aiohttp.ClientError, OSError) as err:
            self._log_grab_failure(
                "error reading a frame from %s - %s: %s",
                url,
                type(err).__name__,
                err,
            )
        return None

    async def _probe_endpoints(self, url: str) -> None:
        """
        Log what the camera host actually serves, once per entity.

        Runs only after a CC2 grab has failed. None of CC2_PROBE_PATHS are
        documented for CC2 firmware; the point is to record what the
        printer answers on so the correct endpoint can be identified from
        a log instead of guessed.
        """
        if self._endpoints_probed:
            return
        self._endpoints_probed = True

        base = URL(url).origin()
        session = async_get_clientsession(self.hass)
        results: list[str] = []
        for path in CC2_PROBE_PATHS:
            candidate = str(base.join(URL(path)))
            try:
                async with session.get(
                    candidate, timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    results.append(
                        f"{path} -> HTTP {resp.status} "
                        f"({resp.headers.get('Content-Type', 'no content-type')})"
                    )
            except (TimeoutError, aiohttp.ClientError, OSError) as err:
                results.append(f"{path} -> {type(err).__name__}: {err}")

        LOGGER.warning(
            "CC2 camera endpoint probe for %s at %s: %s",
            self.entity_id,
            base,
            "; ".join(results),
        )

    async def _async_cc2_camera_image(
        self, width: int | None, height: int | None
    ) -> bytes | None:
        """
        Grab a still from a CC2 chamber camera.

        Serialized on _stream_lock so overlapping snapshot calls cannot
        toggle the printer's video on and off underneath each other, and
        the disable is debounced so consecutive grabs reuse one enabled
        stream.
        """
        async with self._stream_lock:
            # A grab is starting: keep whatever stream is already up.
            self._cancel_pending_disable()
            # A stream that was already enabled (reused inside the debounce
            # window) is already listening, so the readiness probe below is
            # only worth its extra connection right after an enable.
            stream_was_enabled = self._stream_enabled
            if not self._has_active_viewers():
                await self._update_stream_url()
            self._transient_viewers += 1
            try:
                if not self._mjpeg_url:
                    self._log_grab_failure(
                        "no stream URL (printer connected=%s)",
                        self._printer_client.is_connected,
                    )
                    return None
                if self._is_over_capacity():
                    num_connected, max_allowed = self._capacity_counters()
                    self._log_grab_failure(
                        "printer reports video capacity exhausted "
                        "(num_video_stream_connected=%d, "
                        "max_video_stream_allowed=%d)",
                        num_connected,
                        max_allowed,
                    )
                    return None

                url = self._mjpeg_url
                if not stream_was_enabled and not await self._wait_for_stream_ready(
                    url
                ):
                    await self._probe_endpoints(url)
                    return None

                image = await self._fetch_single_frame(url)
                if image is not None:
                    return image

                # Fall back to Home Assistant's own MJPEG image path before
                # giving up — it uses httpx rather than aiohttp, so it can
                # still succeed where the direct grab did not.
                LOGGER.debug(
                    "Direct frame grab returned nothing for %s, "
                    "falling back to the MjpegCamera stream path",
                    self.entity_id,
                )
                try:
                    image = await super().async_camera_image(width=width, height=height)
                except Exception as err:
                    # Logged for diagnosis, then re-raised: Home Assistant
                    # reports a failed snapshot on an exception but returns
                    # silently on None, so swallowing would hide the failure.
                    self._log_grab_failure(
                        "MjpegCamera fallback raised %s: %s",
                        type(err).__name__,
                        err,
                    )
                    raise
                if image is None:
                    self._log_grab_failure(
                        "MjpegCamera fallback also returned no image from %s", url
                    )
                    await self._probe_endpoints(url)
                return image
            finally:
                self._transient_viewers = max(0, self._transient_viewers - 1)
                # Only disable if no other viewers are active. Debounced so a
                # snapshot automation running every few seconds does not
                # re-toggle the printer's video for every frame.
                if not self._has_active_viewers():
                    self._schedule_disable()

    async def async_camera_image(
        self, width: int | None = None, height: int | None = None
    ) -> bytes | None:
        """
        Return a still image from the printer camera.

        Treats the image grab as a transient viewer — ref-counts the
        video stream per ElegooVideoStreamLifecycle: enables the stream
        on the first viewer and disables it when the last viewer
        disconnects. The base MjpegCamera image path reads a single
        frame from a short-lived HTTP connection, which closes when the
        grab completes, so no stream connection is left open afterwards.

        CC2 printers take a separate path that waits for the stream
        server to come up and reports why a grab failed.
        """
        if self._is_cc2:
            return await self._async_cc2_camera_image(width, height)

        # Enable stream if no other viewers are active (check before increment)
        if not self._has_active_viewers():
            await self._update_stream_url()
        self._transient_viewers += 1
        try:
            if not self._mjpeg_url:
                self._log_grab_failure(
                    "no stream URL (printer connected=%s)",
                    self._printer_client.is_connected,
                )
                return None
            if self._is_over_capacity():
                num_connected, max_allowed = self._capacity_counters()
                self._log_grab_failure(
                    "printer reports video capacity exhausted "
                    "(num_video_stream_connected=%d, max_video_stream_allowed=%d)",
                    num_connected,
                    max_allowed,
                )
                return None
            try:
                image = await super().async_camera_image(width=width, height=height)
            except Exception as err:
                # Logged for diagnosis, then re-raised — see the CC2 path.
                self._log_grab_failure(
                    "MjpegCamera raised %s: %s", type(err).__name__, err
                )
                raise
            if image is None:
                self._log_grab_failure(
                    "MjpegCamera returned no image from %s", self._mjpeg_url
                )
            return image
        finally:
            self._transient_viewers = max(0, self._transient_viewers - 1)
            # Only disable if no other viewers are active
            if not self._has_active_viewers():
                await self._disable_stream()

    async def handle_async_mjpeg_stream(
        self, request: web.Request
    ) -> web.StreamResponse:
        """
        Generate an HTTP MJPEG stream from the camera.

        Ref-counted: enables video on first viewer, disables on last.
        """
        # Enable stream if first viewer
        self._cancel_pending_disable()
        if not self._has_active_viewers():
            await self._update_stream_url()
        self._active_mjpeg_streams += 1
        self._last_activity = asyncio.get_running_loop().time()
        try:
            if not self._mjpeg_url or self._is_over_capacity():
                return web.Response(
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                    reason="Stream URL not available",
                )
            return await super().handle_async_mjpeg_stream(request)
        finally:
            # Disable stream if last viewer
            self._last_activity = asyncio.get_running_loop().time()
            self._active_mjpeg_streams = max(0, self._active_mjpeg_streams - 1)
            if not self._has_active_viewers():
                await self._disable_stream()

    async def stream_source(self) -> str | None:
        """
        Return the MJPEG stream source.

        Enables video for native streams (which uses the MJPEG source
        with FFmpeg), tracks the stream, and disables it after
        NATIVE_STREAM_IDLE_TIMEOUT of idle via the idle watchdog.
        """
        if not self._native_stream_active:
            self._cancel_pending_disable()
            await self._update_stream_url()
            if not self._mjpeg_url:
                return None
            self._native_stream_active = True
            self._last_activity = asyncio.get_running_loop().time()
        return self._mjpeg_url
