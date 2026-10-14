"""
Tests for how the CC2 client reads the printer's registration reply.

The client id is stable across restarts, so a quick reload re-registers an
id the printer may still hold. The printer answers "already registered"
then - which means it already knows this client, not that setup failed.

These drive the real connect path - connect_printer, then
_try_connect_with_password, then _register - with the printer's reply fed
through the client's real message handler, so only the broker is faked.
"""

from __future__ import annotations

import asyncio
import json
from typing import Self
from unittest.mock import AsyncMock, MagicMock, patch

import custom_components.elegoo_printer.cc2.client as client_module
from custom_components.elegoo_printer.cc2.client import ElegooCC2Client
from custom_components.elegoo_printer.sdcp.models.printer import Printer

SERIAL = "TEST123"
ACCESS_CODE_HINT = "verify the access code"


class _FakeBroker:
    """
    Stands in for aiomqtt.Client.

    Answers every registration request with `reply` by handing the response
    to the client's own _handle_message, exactly as the listener would.
    """

    def __init__(self, client: ElegooCC2Client, reply: str | None) -> None:
        self._client = client
        self._reply = reply
        self.registrations = 0

    def __call__(self, **_kwargs: object) -> Self:
        return self

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def subscribe(self, *_args: object, **_kwargs: object) -> None:
        return None

    async def publish(self, topic: str, payload: str) -> None:
        if not topic.endswith("/api_register") or self._reply is None:
            return
        self.registrations += 1
        request_id = json.loads(payload)["request_id"]
        response = json.dumps(
            {"client_id": self._client._client_id, "error": self._reply}
        )
        asyncio.get_running_loop().call_soon(
            lambda: asyncio.ensure_future(
                self._client._handle_message(
                    f"elegoo/{SERIAL}/{request_id}/register_response", response
                )
            )
        )


def _make_client(access_code: str | None = "mycode") -> ElegooCC2Client:
    return ElegooCC2Client(
        printer_ip="192.168.1.100",
        serial_number=SERIAL,
        access_code=access_code,
        logger=MagicMock(),
    )


def _make_printer() -> Printer:
    p = Printer()
    p.name = "TestPrinter"
    p.ip_address = "192.168.1.100"
    p.id = SERIAL
    return p


async def _connect(
    client: ElegooCC2Client, reply: str | None
) -> tuple[bool, _FakeBroker]:
    """Run setup against a broker that answers registration with `reply`."""
    broker = _FakeBroker(client, reply)
    with (
        patch.object(client_module.aiomqtt, "Client", broker),
        # The listener, heartbeat and initial data are not what is under test;
        # replies reach the client through _handle_message directly.
        patch.object(client, "_mqtt_listener", AsyncMock()),
        patch.object(client, "_heartbeat_loop", AsyncMock()),
        patch.object(client, "_request_initial_data", AsyncMock()),
    ):
        ok = await client.connect_printer(_make_printer())
        if ok:
            await client.disconnect()
    return ok, broker


def _logged(logger_method: MagicMock) -> str:
    return " ".join(
        (str(c.args[0]) % c.args[1:]) if len(c.args) > 1 else str(c.args[0])
        for c in logger_method.call_args_list
        if c.args
    )


class TestAlreadyRegistered:
    """The printer already knows this client: that is a successful setup."""

    def test_already_registered_setup_succeeds(self) -> None:
        """Register gets "already registered" -> setup succeeds."""

        async def run() -> None:
            client = _make_client()
            ok, broker = await _connect(client, "already registered")
            assert ok is True
            assert broker.registrations == 1
            assert client._register_count == 1
            assert client._registration_rejection is None

        asyncio.run(run())

    def test_reuse_is_logged_at_info_not_error(self) -> None:
        """A reused registration is normal, so it must not look like a fault."""

        async def run() -> None:
            client = _make_client()
            await _connect(client, "already registered")
            info = _logged(client.logger.info)
            assert "still holds our registration" in info
            assert "reusing it" in info
            assert client._client_id in info
            assert "Registration refused" not in _logged(client.logger.error)

        asyncio.run(run())

    def test_tolerates_case_and_whitespace(self) -> None:
        """The reply is undocumented, so do not depend on its exact casing."""

        async def run() -> None:
            ok, _ = await _connect(_make_client(), "  Already Registered ")
            assert ok is True

        asyncio.run(run())

    def test_ok_still_succeeds(self) -> None:
        """The ordinary success reply is unchanged."""

        async def run() -> None:
            ok, _ = await _connect(_make_client(), "ok")
            assert ok is True

        asyncio.run(run())


class TestRefusedRegistration:
    """
    The login worked but the printer refused registration with a reason.

    The access code is evidently right, so the user must see the printer's
    reason rather than advice to check the access code.
    """

    def test_reason_is_passed_through(self) -> None:
        """The printer's own text reaches the error, not the access-code hint."""

        async def run() -> None:
            client = _make_client()
            ok, _ = await _connect(client, "too many clients")
            assert ok is False
            errors = _logged(client.logger.error)
            assert "'too many clients'" in errors
            assert "accepted the login but refused registration" in errors
            assert ACCESS_CODE_HINT not in errors

        asyncio.run(run())

    def test_unknown_reason_is_passed_through(self) -> None:
        """A reason this code has never seen is reported verbatim."""

        async def run() -> None:
            client = _make_client()
            ok, _ = await _connect(client, "fail")
            assert ok is False
            errors = _logged(client.logger.error)
            assert "'fail'" in errors
            assert ACCESS_CODE_HINT not in errors

        asyncio.run(run())

    def test_does_not_cycle_fallback_passwords(self) -> None:
        """
        A refusal after a successful login stops the password loop.

        With no access code configured the client would otherwise go on to
        try the fallback passwords - each one another connection against a
        printer with a small client cap, and none able to change its answer.
        """

        async def run() -> None:
            client = _make_client(access_code=None)
            ok, broker = await _connect(client, "too many clients")
            assert ok is False
            assert broker.registrations == 1

        asyncio.run(run())

    def test_refused_session_is_disconnected(self) -> None:
        """Stopping the loop must not leave the refused MQTT session open."""

        async def run() -> None:
            client = _make_client(access_code=None)
            ok, _ = await _connect(client, "too many clients")
            assert ok is False
            assert client.mqtt_client is None
            assert "registration refused: too many clients" in (
                client._last_disconnect_reason
            )

        asyncio.run(run())


class TestNoReply:
    """No reply at all is a different failure and keeps its existing advice."""

    def test_timeout_still_suggests_the_access_code(self) -> None:
        """Only a reply with a reason clears the access code as a suspect."""

        async def run() -> None:
            client = _make_client()
            with patch.object(client_module, "CC2_REGISTRATION_TIMEOUT", 0.05):
                ok, _ = await _connect(client, None)
            assert ok is False
            assert client._registration_rejection is None
            assert ACCESS_CODE_HINT in _logged(client.logger.error)

        asyncio.run(run())
