"""
同期 BLE クライアント

ble_server.py が提供する Unix ドメインソケット (/tmp/ble_server.sock) に
JSON コマンドを送信し、結果を返す。

ehr_input.py など同期コードから BLE 操作を呼び出すために使用する。
BLE 接続はサーバー側で保持されるため、接続コストは呼び出し側では発生しない。
"""

import json
import socket
import time

SOCKET_PATH = "/tmp/ble_server.sock"


class BLEClient:
    """Unix ソケット経由で ble_server.py にコマンドを送るクライアント"""

    @staticmethod
    def _normalize_key_name(key: str) -> str:
        normalized = key.strip().lower()
        if normalized == "escape":
            return "esc"
        return normalized

    def _send(self, req: dict, timeout: float = 30.0) -> dict:
        """1コマンドを送信して結果を返す"""
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(SOCKET_PATH)
            s.sendall((json.dumps(req) + "\n").encode())
            data = b""
            while not data.endswith(b"\n"):
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
        return json.loads(data.strip())

    def is_server_running(self) -> bool:
        """サーバーが起動して BLE 接続済みかどうかを確認"""
        try:
            result = self._send({"cmd": "status"})
            return result.get("ok", False) and result.get("connected", False)
        except (ConnectionRefusedError, FileNotFoundError):
            return False

    def switch_to_mouse_mode(self) -> bool:
        return self._send({"cmd": "switch_to_mouse_mode"})["ok"]

    def switch_to_keyboard_mode(self) -> bool:
        return self._send({"cmd": "switch_to_keyboard_mode"})["ok"]

    def move_mouse_to_position(self, x: int, y: int) -> bool:
        return self._send({"cmd": "move_mouse_to_position", "x": x, "y": y})["ok"]

    def move_mouse_absolute(self, x: int, y: int) -> bool:
        return self._send({"cmd": "move_mouse_absolute", "x": x, "y": y})["ok"]

    def move_mouse(self, x: int, y: int) -> bool:
        return self._send({"cmd": "move_mouse", "x": x, "y": y})["ok"]

    def click(self) -> bool:
        return self._send({"cmd": "click"})["ok"]

    def double_click(self) -> bool:
        return self._send({"cmd": "double_click"})["ok"]

    def right_click(self) -> bool:
        return self._send({"cmd": "right_click"})["ok"]

    def scroll(self, amount: int) -> bool:
        return self._send({"cmd": "scroll", "amount": amount})["ok"]

    def type_text(self, text: str) -> bool:
        if not text.isascii():
            bad = [(i, c, f"U+{ord(c):04X}") for i, c in enumerate(text) if ord(c) > 127]
            raise ValueError(
                f"type_text received non-ASCII characters {bad}; "
                f"these would produce unpredictable HID key events on the ESP32"
            )
        return self._send({"cmd": "type_text", "text": text})["ok"]

    def press_key(self, key: str) -> bool:
        return self._send({"cmd": "press_key", "key": self._normalize_key_name(key)})["ok"]

    def press_ime_toggle(self) -> bool:
        """IME 半角/全角トグルキーを送る"""
        return self.press_key("zenkaku")

    def send_command(self, command: str) -> bool:
        return self._send({"cmd": "send_command", "command": command})["ok"]

    def clear_editor_document(self, max_chars: int = 200, delay: float = 0.12) -> None:
        """エディターの全テキストを削除する。

        クリック→フォーカス確保→Escape（IMEキャンセル）→ctrl+End→Backspace×max_chars の順で実行する。
        ctrl+a による全選択はこのエディターでは動作しないため、末尾から1文字ずつ削除する方式を使用する。
        """
        self.click()
        time.sleep(0.5)
        self.press_key("escape")
        time.sleep(0.3)
        self.press_key("ctrl+end")
        time.sleep(0.5)
        for _ in range(max_chars):
            self.press_key("backspace")
            time.sleep(delay)
