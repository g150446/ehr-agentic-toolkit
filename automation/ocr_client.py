"""Client for the resident PaddleOCR server."""

from __future__ import annotations

import json
import os
import socket
import tempfile
from typing import Optional

import cv2


OCR_SERVER_SOCKET_PATH = os.getenv("OCR_SERVER_SOCKET_PATH", "/tmp/paddle_ocr_server.sock")
OCR_SERVER_TIMEOUT = float(os.getenv("OCR_SERVER_TIMEOUT", "120"))


class OCRServerError(RuntimeError):
    """Raised when the OCR server cannot fulfill a request."""


class OCRClient:
    """Unix socket client for automation.ocr_server."""

    def __init__(
        self,
        socket_path: str = OCR_SERVER_SOCKET_PATH,
        timeout: float = OCR_SERVER_TIMEOUT,
    ):
        self.socket_path = socket_path
        self.timeout = timeout

    def _send(self, req: dict) -> dict:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout)
                s.connect(self.socket_path)
                s.sendall((json.dumps(req) + "\n").encode())
                data = b""
                while not data.endswith(b"\n"):
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    data += chunk
        except (ConnectionRefusedError, FileNotFoundError, socket.timeout, OSError) as exc:
            raise OCRServerError(
                "OCR サーバーに接続できませんでした。"
                " 別ターミナルで ./scripts/start_ocr_server.sh を起動してください。"
            ) from exc

        if not data:
            raise OCRServerError("OCR サーバーから応答がありませんでした")

        result = json.loads(data.strip())
        if not result.get("ok", False):
            raise OCRServerError(result.get("error", "OCR サーバーエラー"))
        return result

    def status(self) -> dict:
        return self._send({"cmd": "status"})

    def is_server_running(self) -> bool:
        try:
            result = self.status()
        except OCRServerError:
            return False
        return result.get("ready", False)

    def ocr_image_path(self, image_path: str, languages: list[str]) -> list[tuple]:
        result = self._send(
            {
                "cmd": "ocr_image_path",
                "image_path": image_path,
                "languages": languages,
            }
        )
        return [
            (bbox, text, float(conf))
            for bbox, text, conf in result.get("results", [])
        ]


def request_ocr(
    image,
    *,
    languages: Optional[list[str]] = None,
    socket_path: str = OCR_SERVER_SOCKET_PATH,
    timeout: float = OCR_SERVER_TIMEOUT,
) -> list[tuple]:
    """Send an image to the resident OCR server and return normalized OCR tuples."""
    actual_languages = languages or ["ja", "en"]
    client = OCRClient(socket_path=socket_path, timeout=timeout)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        image_path = f.name
    try:
        if not cv2.imwrite(image_path, image):
            raise OCRServerError(f"一時OCR画像を書き込めませんでした: {image_path}")
        return client.ocr_image_path(image_path, actual_languages)
    finally:
        if os.path.exists(image_path):
            os.unlink(image_path)
