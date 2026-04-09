import json
import socket
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from automation import ehr_input
from automation.ollama_segmentation import (
    OllamaSegmentationError,
    parse_segment_response,
    segment_japanese_text_with_ollama,
)
from automation.ollama_segment_probe import main as probe_main


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_parse_segment_response_extracts_json_array():
    content = '了解です [{"text": "肺炎", "romaji": "haien"}] 以上です'

    segments = parse_segment_response(content)

    assert segments == [{"text": "肺炎", "romaji": "haien"}]


def test_segment_japanese_text_with_ollama_returns_segments(monkeypatch):
    def fake_urlopen(req, timeout):
        assert timeout == 12
        assert req.full_url == "http://localhost:11434/api/generate"
        payload = json.loads(req.data.decode())
        assert payload["model"] == "gemma4:e2b"
        assert payload["stream"] is False
        assert "入力: 肺炎に対して抗菌薬による治療を行う" in payload["prompt"]
        return _FakeResponse(
            {
                "response": '[{"text": "肺炎", "romaji": "haien"}, '
                '{"text": "に対して", "romaji": "nitaishite"}]'
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    raw_content, segments = segment_japanese_text_with_ollama(
        "肺炎に対して抗菌薬による治療を行う",
        timeout=12,
    )

    assert "肺炎" in raw_content
    assert segments == [
        {"text": "肺炎", "romaji": "haien"},
        {"text": "に対して", "romaji": "nitaishite"},
    ]


def test_segment_japanese_text_with_ollama_wraps_timeout(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.URLError(socket.timeout("timed out"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(OllamaSegmentationError, match="タイムアウト"):
        segment_japanese_text_with_ollama("肺炎に対して抗菌薬による治療を行う")


def test_probe_main_prints_segments(monkeypatch, capsys):
    monkeypatch.setattr(
        "automation.ollama_segment_probe.segment_japanese_text_with_ollama",
        lambda text: (
            '[{"text": "肺炎", "romaji": "haien"}]',
            [{"text": "肺炎", "romaji": "haien"}],
        ),
    )

    exit_code = probe_main(["肺炎に対して抗菌薬による治療を行う"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "分割結果:" in captured.out
    assert "'肺炎' (haien)" in captured.out


def test_ehr_input_main_reports_segmentation_error(monkeypatch, capsys):
    monkeypatch.setattr(
        ehr_input,
        "type_japanese_sentence",
        lambda text: (_ for _ in ()).throw(OllamaSegmentationError("timed out")),
    )

    exit_code = ehr_input.main(["肺炎に対して抗菌薬による治療を行う"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Ollama文節分割エラー: timed out" in captured.out
    assert "automation.ollama_segment_probe" in captured.out
