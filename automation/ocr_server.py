"""Resident PaddleOCR server."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import signal
from datetime import datetime

import cv2

from automation.config import AutomationConfig
from automation.screen_analyzer import load_paddleocr_reader, run_ocr


def _configure_paddle_runtime(config: AutomationConfig) -> dict:
    """Try to configure the best available Paddle runtime and report the result."""
    import paddle

    requested = config.ocr_server_device
    actual = "cpu"
    note = ""

    if requested == "cpu":
        paddle.set_device("cpu")
        note = "CPU 強制"
    elif requested == "auto":
        if paddle.device.is_compiled_with_cuda():
            paddle.set_device("gpu")
            actual = "gpu"
            note = "CUDA 利用"
        else:
            paddle.set_device("cpu")
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                note = "この PaddlePaddle ビルドでは Apple GPU / MPS は未対応のため CPU を使用"
            else:
                note = "CPU 利用"
    else:
        paddle.set_device(requested)
        actual = requested
        note = f"{requested} 強制"

    return {"requested": requested, "actual": actual, "note": note}


async def dispatch(state: dict, req: dict) -> dict:
    cmd = req.get("cmd")
    try:
        if cmd == "status":
            return {
                "ok": True,
                "ready": True,
                "socket_path": state["socket_path"],
                "languages": state["languages"],
                "device": state["device"],
            }

        if cmd == "ocr_image_path":
            image_path = req.get("image_path")
            languages = req.get("languages") or state["languages"]
            if not image_path:
                return {"ok": False, "error": "image_path is required"}
            image = cv2.imread(image_path)
            if image is None:
                return {"ok": False, "error": f"画像を読み込めませんでした: {image_path}"}
            reader = load_paddleocr_reader(languages)
            results = run_ocr(reader, image)
            return {"ok": True, "results": results}

        return {"ok": False, "error": f"Unknown command: {cmd}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def handle_client(state: dict, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                req = json.loads(line)
            except json.JSONDecodeError as exc:
                result = {"ok": False, "error": f"Invalid JSON: {exc}"}
            else:
                result = await dispatch(state, req)
            writer.write((json.dumps(result, ensure_ascii=False) + "\n").encode())
            await writer.drain()
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def main() -> None:
    config = AutomationConfig()
    runtime = _configure_paddle_runtime(config)

    print(f"OCR サーバー初期化中... device={runtime['actual']} ({runtime['note']})")
    load_paddleocr_reader(config.ocr_languages)
    print(f"PaddleOCR preload 完了: languages={config.ocr_languages}")

    if os.path.exists(config.ocr_server_socket_path):
        os.unlink(config.ocr_server_socket_path)

    state = {
        "socket_path": config.ocr_server_socket_path,
        "languages": config.ocr_languages,
        "device": runtime,
    }
    server = await asyncio.start_unix_server(
        lambda r, w: handle_client(state, r, w),
        path=config.ocr_server_socket_path,
    )

    print(f"OCR サーバー起動: {config.ocr_server_socket_path}")
    print("停止するには Ctrl+C を押してください。")

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _shutdown():
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] OCR サーバーを停止します...")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _shutdown)

    async with server:
        await stop_event.wait()

    if os.path.exists(config.ocr_server_socket_path):
        os.unlink(config.ocr_server_socket_path)
    print("OCR サーバー終了")


if __name__ == "__main__":
    asyncio.run(main())
