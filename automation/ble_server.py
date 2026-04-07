"""
BLE 常駐サーバー

起動時に ESP32 BLE デバイスへ接続し、Unix ドメインソケット
(/tmp/ble_server.sock) 経由でマウス・キーボードコマンドを受け付ける。

使用方法:
    python -m automation.ble_server

プロトコル: 改行区切り JSON (1リクエスト1レスポンス)
  リクエスト例: {"cmd": "click"}
  レスポンス例: {"ok": true}
"""

import asyncio
import json
import os
import signal
import sys

from automation.ble_controller import BLEController
from automation.config import AutomationConfig

SOCKET_PATH = "/tmp/ble_server.sock"


async def dispatch(ble: BLEController, req: dict) -> dict:
    """コマンドを対応する BLEController メソッドにルーティング"""
    cmd = req.get("cmd")
    try:
        if cmd == "status":
            return {
                "ok": True,
                "connected": ble.is_connected(),
                "address": ble.device_address,
            }
        if cmd == "connect":
            ok = await ble.connect(req.get("timeout", 15.0))
            return {"ok": ok}
        if cmd == "disconnect":
            await ble.disconnect()
            return {"ok": True}
        if cmd == "switch_to_mouse_mode":
            return {"ok": await ble.switch_to_mouse_mode()}
        if cmd == "switch_to_keyboard_mode":
            return {"ok": await ble.switch_to_keyboard_mode()}
        if cmd == "move_mouse_to_position":
            return {"ok": await ble.move_mouse_to_position(req["x"], req["y"])}
        if cmd == "move_mouse_absolute":
            return {"ok": await ble.move_mouse_absolute(req["x"], req["y"])}
        if cmd == "move_mouse":
            return {"ok": await ble.move_mouse(req["x"], req["y"])}
        if cmd == "click":
            return {"ok": await ble.click()}
        if cmd == "right_click":
            return {"ok": await ble.right_click()}
        if cmd == "scroll":
            return {"ok": await ble.scroll(req["amount"])}
        if cmd == "type_text":
            return {"ok": await ble.type_text(req["text"])}
        if cmd == "press_key":
            return {"ok": await ble.press_key(req["key"])}
        if cmd == "send_command":
            return {"ok": await ble.send_command(req["command"])}
        return {"ok": False, "error": f"Unknown command: {cmd}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def handle_client(ble: BLEController, reader: asyncio.StreamReader,
                         writer: asyncio.StreamWriter) -> None:
    """クライアント1接続のリクエスト/レスポンスループ"""
    peer = writer.get_extra_info("peername", "unknown")
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                req = json.loads(line)
            except json.JSONDecodeError as e:
                result = {"ok": False, "error": f"Invalid JSON: {e}"}
            else:
                result = await dispatch(ble, req)
            writer.write((json.dumps(result) + "\n").encode())
            await writer.drain()
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def main() -> None:
    config = AutomationConfig()
    ble = BLEController(
        device_name=config.esp32_device_name,
        service_uuid=config.ble_service_uuid,
        rx_char_uuid=config.ble_rx_char_uuid,
        tx_char_uuid=config.ble_tx_char_uuid,
    )

    print(f"BLE デバイス '{config.esp32_device_name}' に接続中...")
    connected = await ble.connect(timeout=15.0)
    if connected:
        print(f"接続成功: {ble.device_address}")
    else:
        print("接続失敗。サーバーは起動しますが BLE コマンドは失敗します。")
        print("再接続するには {'cmd': 'connect'} を送信してください。")

    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    server = await asyncio.start_unix_server(
        lambda r, w: handle_client(ble, r, w),
        path=SOCKET_PATH,
    )
    print(f"BLE サーバー起動: {SOCKET_PATH}")
    print("停止するには Ctrl+C を押してください。")

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _shutdown():
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _shutdown)

    async with server:
        await stop_event.wait()

    print("\nシャットダウン中...")
    if ble.is_connected():
        await ble.disconnect()
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    print("BLE サーバー終了")


if __name__ == "__main__":
    asyncio.run(main())
