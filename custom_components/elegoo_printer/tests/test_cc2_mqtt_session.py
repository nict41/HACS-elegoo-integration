"""
Tests for the CC2 MQTT session identity and telemetry.

The printer registers clients statefully and caps how many it holds, so
the integration's connect/register/disconnect behaviour is a suspect for
the chamber camera dying after a couple of hours. These cover the stable
client identity and the session counters used to investigate it.
"""

import asyncio
import hashlib
from unittest.mock import MagicMock, patch

from custom_components.elegoo_printer.cc2.client import ElegooCC2Client


def _run(coro):
    """Run an async coroutine to completion (fresh loop)."""
    asyncio.run(coro)


def _client(serial: str = "SN12345678") -> ElegooCC2Client:
    """Build a CC2 client without connecting to anything."""
    return ElegooCC2Client(
        printer_ip="192.168.8.128",
        serial_number=serial,
        access_code="secret",
        logger=MagicMock(),
    )


class TestStableClientId:
    """The MQTT client identity must not change between runs."""

    def test_same_serial_gives_same_id(self) -> None:
        """A restart must not present the printer with a new client."""
        assert _client()._client_id == _client()._client_id

    def test_different_serials_differ(self) -> None:
        """Two printers must not collide on one identity."""
        assert _client("AAA")._client_id != _client("BBB")._client_id

    def test_matches_web_interface_format(self) -> None:
        """Exactly 10 chars, "0cli" prefix, hex tail."""
        client_id = _client()._client_id
        assert len(client_id) == 10
        assert client_id.startswith("0cli")
        int(client_id[4:], 16)

    def test_derived_from_serial(self) -> None:
        """The identity is a pure function of the serial."""
        expected = "0cli" + hashlib.sha256(b"SN12345678").hexdigest()[:6]
        assert _client()._client_id == expected

    def test_falls_back_to_ip_without_serial(self) -> None:
        """A client with no serial still gets a stable identity."""
        client = ElegooCC2Client(
            printer_ip="192.168.8.128",
            serial_number="",
            access_code=None,
            logger=MagicMock(),
        )
        expected = "0cli" + hashlib.sha256(b"192.168.8.128").hexdigest()[:6]
        assert client._client_id == expected


class TestSessionTelemetry:
    """Counters that show how hard the printer's MQTT stack is worked."""

    def test_counters_start_at_zero(self) -> None:
        """A fresh client has done nothing yet."""
        client = _client()
        assert client._connect_count == 0
        assert client._register_count == 0
        assert client._disconnect_count == 0

    def test_disconnect_records_reason_and_counts(self) -> None:
        """Every disconnect is counted and carries why it happened."""

        async def run() -> None:
            client = _client()
            client.mqtt_client = None
            await client.disconnect(reason="heartbeat timeout, no PONG in 70s")
            assert client._disconnect_count == 1
            assert client._last_disconnect_reason == (
                "heartbeat timeout, no PONG in 70s"
            )
            logged = str(client.logger.info.call_args_list)
            assert "CC2MQTT disconnecting" in logged

        _run(run())

    def test_disconnect_defaults_to_requested(self) -> None:
        """An ordinary teardown is distinguishable from a failure."""

        async def run() -> None:
            client = _client()
            client.mqtt_client = None
            await client.disconnect()
            assert client._last_disconnect_reason == "requested"

        _run(run())

    def test_session_stats_string(self) -> None:
        """The summary carries every counter the analysis needs."""
        client = _client()
        client._connect_count = 3
        client._register_count = 3
        client._disconnect_count = 2
        client._video_command_count = 287
        client._last_disconnect_reason = "heartbeat error"
        stats = client.mqtt_session_stats()
        assert "connects=3" in stats
        assert "registrations=3" in stats
        assert "disconnects=2" in stats
        assert "video_commands=287" in stats
        assert "heartbeat error" in stats

    def test_uptime_counts_from_session_start(self) -> None:
        """Uptime reflects the live session, not wall clock since boot."""
        client = _client()
        with patch("time.time", return_value=1000.0):
            client._session_started = 940.0
            assert "uptime=60s" in client.mqtt_session_stats()

    def test_uptime_zero_when_disconnected(self) -> None:
        """No session, no uptime."""
        client = _client()
        client._session_started = None
        assert "uptime=0s" in client.mqtt_session_stats()


class TestVideoCommandCounting:
    """Method-1042 sends are the integration's camera write footprint."""

    def test_each_send_is_counted_and_logged(self) -> None:
        """Enable and disable both count, and say which they were."""

        async def run() -> None:
            client = _client()
            with patch.object(client, "_send_command", return_value=None):
                await client.set_printer_video_stream(enable=True)
                await client.set_printer_video_stream(enable=False)
            assert client._video_command_count == 2
            logged = str(client.logger.info.call_args_list)
            assert "ENABLE" in logged
            assert "DISABLE" in logged

        _run(run())
