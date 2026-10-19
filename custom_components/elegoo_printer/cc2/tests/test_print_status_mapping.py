"""Tests for mapping CC2 machine/sub-status pairs to the print status."""

from __future__ import annotations

import pytest

from custom_components.elegoo_printer.cc2.const import (
    CC2_STATUS_AUTO_LEVELING,
    CC2_STATUS_HOMING,
    CC2_STATUS_IDLE,
    CC2_STATUS_PRINTING,
    CC2_SUBSTATUS_AUTO_LEVELING_COMPLETED,
    CC2_SUBSTATUS_HOMING,
    CC2_SUBSTATUS_HOMING_COMPLETED,
    CC2_SUBSTATUS_PRINTING,
    CC2_SUBSTATUS_PRINTING_COMPLETED,
)
from custom_components.elegoo_printer.cc2.models import CC2StatusMapper
from custom_components.elegoo_printer.sdcp.models.enums import ElegooPrintStatus


def _print_status(status: int, sub_status: int) -> ElegooPrintStatus:
    data = {"machine_status": {"status": status, "sub_status": sub_status}}
    return CC2StatusMapper.map_status(data).print_info.status


@pytest.mark.parametrize(
    ("status", "sub_status"),
    [
        # Captured on a CC2 (fw 02.01.01.07) after a standalone home command:
        # back to idle, with "homing completed" still reported.
        (CC2_STATUS_IDLE, CC2_SUBSTATUS_HOMING_COMPLETED),
        (CC2_STATUS_HOMING, CC2_SUBSTATUS_HOMING),
        (CC2_STATUS_AUTO_LEVELING, CC2_SUBSTATUS_AUTO_LEVELING_COMPLETED),
        (CC2_STATUS_IDLE, CC2_SUBSTATUS_AUTO_LEVELING_COMPLETED),
    ],
)
def test_homing_or_leveling_outside_a_print_is_idle(  # noqa: D103
    status: int, sub_status: int
) -> None:
    assert _print_status(status, sub_status) == ElegooPrintStatus.IDLE


@pytest.mark.parametrize(
    ("sub_status", "expected"),
    [
        (CC2_SUBSTATUS_HOMING, ElegooPrintStatus.PRINTING),
        (CC2_SUBSTATUS_HOMING_COMPLETED, ElegooPrintStatus.PRINTING),
        (CC2_SUBSTATUS_AUTO_LEVELING_COMPLETED, ElegooPrintStatus.LEVELING),
    ],
)
def test_homing_or_leveling_during_a_print_keeps_its_mapping(  # noqa: D103
    sub_status: int, expected: ElegooPrintStatus
) -> None:
    assert _print_status(CC2_STATUS_PRINTING, sub_status) == expected


def test_other_sub_states_are_unaffected_by_machine_status() -> None:  # noqa: D103
    assert (
        _print_status(CC2_STATUS_IDLE, CC2_SUBSTATUS_PRINTING_COMPLETED)
        == ElegooPrintStatus.COMPLETE
    )
    assert (
        _print_status(CC2_STATUS_PRINTING, CC2_SUBSTATUS_PRINTING)
        == ElegooPrintStatus.PRINTING
    )
