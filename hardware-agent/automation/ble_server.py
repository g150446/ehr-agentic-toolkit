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
from datetime import datetime

from automation.ble_controller import BLEController
from automation.config import AutomationConfig

SOCKET_PATH = "/tmp/ble_server.sock"
BLE_HEALTH_CHECK_INTERVAL = 3.0
BLE_KEEPALIVE_INTERVAL = 10.0  # Send keepalive every 10s to prevent idle disconnect


def print_connection_failure(ble: BLEController) -> None:
    """Print the most useful BLE failure details for terminal users."""
    detail = ble.get_last_error()
    print("接続失敗。start_ble_server.sh が自動で再起動します。")
    if detail:
        print(f"詳細: {detail}")
        if "not authorized" in detail.lower():
            print("macOS の「プライバシーとセキュリティ」→「Bluetooth」で、このターミナルアプリの Bluetooth 権限を許可してください。")


async def dispatch(ble: BLEController, ble_lock: asyncio.Lock, req: dict) -> dict:
    """コマンドを対応する BLEController メソッドにルーティング

    BLE write 操作は asyncio.Lock で直列化する。
    複数の asyncio タスク（handle_client + monitor_ble_connection keepalive）が
    並行して write_gatt_char を呼ぶと macOS CoreBluetooth 内部でブロックが発生するため。
    status / connect / disconnect はロック不要。
    """
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
        async with ble_lock:
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
            if cmd == "double_click":
                return {"ok": await ble.double_click()}
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
            if cmd == "alt_tab":
                return {"ok": await ble.send_command("key:alt_tab")}
            return {"ok": False, "error": f"Unknown command: {cmd}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def handle_client(ble: BLEController, ble_lock: asyncio.Lock,
                         reader: asyncio.StreamReader,
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
                result = await dispatch(ble, ble_lock, req)
            writer.write((json.dumps(result) + "\n").encode())
            await writer.drain()
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def disconnect_and_exit_loop(
    disconnect_event: asyncio.Event,
    stop_event: asyncio.Event,
) -> None:
    """BLE 切断イベントを待ち、stop_event をセットしてサーバーをシャットダウンする。

    再接続は start_ble_server.sh の再起動ループに委ねる。
    プロセスを再起動することで毎回クリーンな BLE 接続状態を確保する。
    """
    await disconnect_event.wait()
    if stop_event.is_set():
        return
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] BLE 切断を検知。サーバーをシャットダウンします（start_ble_server.sh が再起動します）...")
    stop_event.set()


async def monitor_ble_connection(
    ble: BLEController,
    ble_lock: asyncio.Lock,
    disconnect_event: asyncio.Event,
    stop_event: asyncio.Event,
    interval: float = BLE_HEALTH_CHECK_INTERVAL,
) -> None:
    """BLE 接続状態を定期監視し、callback が来ない切断も検知する。
    
    キープアライブとして定期的に status コマンドを送信し、
    アイドルによる切断を防ぐ。
    """
    last_keepalive = asyncio.get_event_loop().time()
    while not stop_event.is_set():
        await asyncio.sleep(interval)
        if stop_event.is_set():
            return
        if not ble.is_connected():
            if disconnect_event.is_set():
                return
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] BLE 接続監視で切断を検知しました。")
            disconnect_event.set()
            return
        # Send keepalive to prevent idle disconnect
        now = asyncio.get_event_loop().time()
        if now - last_keepalive >= BLE_KEEPALIVE_INTERVAL:
            try:
                async with ble_lock:
                    await ble.send_command("mode:mouse")
            except Exception:
                pass
            last_keepalive = now


async def main() -> None:
    config = AutomationConfig()
    ble = BLEController(
        device_name=config.esp32_device_name,
        service_uuid=config.ble_service_uuid,
        rx_char_uuid=config.ble_rx_char_uuid,
        tx_char_uuid=config.ble_tx_char_uuid,
    )

    loop = asyncio.get_running_loop()
    disconnect_event = asyncio.Event()
    ble_lock = asyncio.Lock()

    def on_ble_disconnect(client):
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] BLE デバイスが切断されました。")
        loop.call_soon_threadsafe(disconnect_event.set)

    print(f"BLE デバイス '{config.esp32_device_name}' に接続中...")
    connected = await ble.connect(timeout=15.0, disconnected_callback=on_ble_disconnect)
    if connected:
        print(f"接続成功: {ble.device_address}")
    else:
        print_connection_failure(ble)
        sys.exit(1)

    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    server = await asyncio.start_unix_server(
        lambda r, w: handle_client(ble, ble_lock, r, w),
        path=SOCKET_PATH,
    )
    print(f"BLE サーバー起動: {SOCKET_PATH}")
    print("停止するには Ctrl+C を押してください。")

    stop_event = asyncio.Event()

    def _shutdown():
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _shutdown)

    async with server:
        disconnect_task = asyncio.create_task(
            disconnect_and_exit_loop(disconnect_event, stop_event)
        )
        watchdog_task = asyncio.create_task(
            monitor_ble_connection(ble, ble_lock, disconnect_event, stop_event)
        )
        await stop_event.wait()
        disconnect_task.cancel()
        watchdog_task.cancel()
        try:
            await disconnect_task
        except asyncio.CancelledError:
            pass
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass

    print("\nシャットダウン中...")
    if ble.is_connected():
        await ble.disconnect()
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    print("BLE サーバー終了")


if __name__ == "__main__":
    asyncio.run(main())
