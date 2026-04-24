from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from automation.ble_client import BLEClient
from automation.ble_controller import BLEController
from automation.ble_test_cli import BLETestShell


@pytest.mark.asyncio
async def test_ble_controller_press_key_normalizes_escape_alias():
    controller = BLEController(
        device_name="BLE Mouse & Keyboard",
        service_uuid="service",
        rx_char_uuid="rx",
        tx_char_uuid="tx",
    )
    controller.send_command = AsyncMock(return_value=True)

    ok = await controller.press_key("escape")

    assert ok is True
    controller.send_command.assert_awaited_once_with("key:esc")


def test_ble_client_press_key_normalizes_escape_alias(monkeypatch):
    sent = {}
    client = BLEClient()

    def fake_send(req, timeout=30.0):
        sent["req"] = req
        return {"ok": True}

    monkeypatch.setattr(client, "_send", fake_send)

    client.press_key("escape")

    assert sent["req"] == {"cmd": "press_key", "key": "esc"}


def test_ble_test_shell_press_accepts_escape_alias(monkeypatch):
    pressed = []

    class FakeRunner:
        def __init__(self, config):
            pass

        def is_connected(self):
            return True

        def press_key(self, key):
            pressed.append(key)
            return True

        def disconnect(self):
            return None

        def cleanup(self):
            return None

    monkeypatch.setattr("automation.ble_test_cli.AsyncBLERunner", FakeRunner)
    shell = BLETestShell(SimpleNamespace())

    shell.do_press("escape")
    shell.do_esc("")

    assert pressed == ["esc", "esc"]
