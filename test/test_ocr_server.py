import asyncio

import cv2
import numpy as np

import automation.ocr_server as ocr_server


def test_dispatch_status_returns_ready_state():
    state = {
        "socket_path": "/tmp/test-ocr.sock",
        "languages": ["ja", "en"],
        "device": {"requested": "auto", "actual": "cpu", "note": "CPU"},
    }

    result = asyncio.run(ocr_server.dispatch(state, {"cmd": "status"}))

    assert result["ok"] is True
    assert result["ready"] is True
    assert result["socket_path"] == "/tmp/test-ocr.sock"


def test_dispatch_ocr_image_path_uses_paddleocr_loader(monkeypatch, tmp_path):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image_path = tmp_path / "ocr.png"
    cv2.imwrite(str(image_path), image)

    monkeypatch.setattr(ocr_server, "load_paddleocr_reader", lambda languages: "reader")
    monkeypatch.setattr(
        ocr_server,
        "run_ocr",
        lambda reader, actual_image: [([[1, 2], [3, 2], [3, 4], [1, 4]], "患者検索", 0.98)],
    )

    state = {
        "socket_path": "/tmp/test-ocr.sock",
        "languages": ["ja", "en"],
        "device": {"requested": "auto", "actual": "cpu", "note": "CPU"},
    }

    result = asyncio.run(
        ocr_server.dispatch(
            state,
            {"cmd": "ocr_image_path", "image_path": str(image_path), "languages": ["ja", "en"]},
        )
    )

    assert result["ok"] is True
    assert result["results"][0][1] == "患者検索"
