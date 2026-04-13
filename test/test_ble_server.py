import asyncio

import pytest

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

    task = asyncio.create_task(
        monitor_ble_connection(FakeBLE(), disconnect_event, stop_event, interval=0.01)
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

    await asyncio.wait_for(
        monitor_ble_connection(FakeBLE(), disconnect_event, stop_event, interval=0.01),
        timeout=0.2,
    )

    assert not disconnect_event.is_set()
