import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from automation.ble_controller import BLEController
from automation.ble_server import monitor_ble_connection


@pytest.mark.asyncio
async def test_monitor_ble_connection_sets_disconnect_event_when_link_drops():
    class FakeBLE:
        def __init__(self):
            self.calls = 0

        def is_connected(self):
            self.calls += 1
            return self.calls < 2

    disconnect_event = asyncio.Event()
    stop_event = asyncio.Event()
    ble_lock = asyncio.Lock()

    task = asyncio.create_task(
        monitor_ble_connection(FakeBLE(), ble_lock, disconnect_event, stop_event, interval=0.01)
    )

    await asyncio.wait_for(disconnect_event.wait(), timeout=0.2)
    stop_event.set()
    await asyncio.wait_for(task, timeout=0.2)


@pytest.mark.asyncio
async def test_monitor_ble_connection_exits_cleanly_when_stopped():
    class FakeBLE:
        def is_connected(self):
            return True

    disconnect_event = asyncio.Event()
    stop_event = asyncio.Event()
    stop_event.set()
    ble_lock = asyncio.Lock()

    await asyncio.wait_for(
        monitor_ble_connection(FakeBLE(), ble_lock, disconnect_event, stop_event, interval=0.01),
        timeout=0.2,
    )

    assert not disconnect_event.is_set()


@pytest.mark.asyncio
async def test_ble_controller_records_scan_authorization_error():
    controller = BLEController(
        device_name="BLE Mouse & Keyboard",
        service_uuid="service",
        rx_char_uuid="rx",
        tx_char_uuid="tx",
    )

    with patch("automation.ble_controller.BleakScanner.discover", new=AsyncMock(side_effect=Exception("BLE is not authorized - check macOS privacy settings"))):
        ok = await controller.connect(timeout=0.01)

    assert not ok
    assert controller.get_last_error() == "BLE scan failed: BLE is not authorized - check macOS privacy settings"
