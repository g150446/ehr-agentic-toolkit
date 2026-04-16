from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import automation.ehr_input as ehr_input
from automation.mlx_vlm_segmentation import MlxVlmSegmentationError


def test_resolve_text_argument_reads_utf8_file(tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("肺炎に対する治療\n", encoding="utf-8")

    assert ehr_input._resolve_text_argument(str(note)) == "肺炎に対する治療"


def test_run_cli_uses_file_contents_for_direct_input(monkeypatch, tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("肺炎に対する治療", encoding="utf-8")
    events = []

    monkeypatch.setattr(ehr_input, "_input_resolved_text", lambda text, **kw: events.append(text))

    assert ehr_input._run_cli([str(note)]) == 0
    assert events == ["肺炎に対する治療"]


def test_run_cli_uses_file_contents_for_open_test(monkeypatch, tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("COVID-19の感染を確認した", encoding="utf-8")
    events = []

    monkeypatch.setattr(ehr_input, "open_test_patient_chart", lambda: events.append("open"))
    monkeypatch.setattr(ehr_input, "_input_resolved_text", lambda text, **kw: events.append(text))

    assert ehr_input._run_cli(["open test", str(note)]) == 0
    assert events == ["open", "COVID-19の感染を確認した"]


def test_run_cli_prioritizes_command_over_same_named_file(monkeypatch, tmp_path):
    command_name = "click history 20260408"
    Path(tmp_path / command_name).write_text("dummy", encoding="utf-8")
    events = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ehr_input, "click_history", lambda date_str: events.append(date_str))

    assert ehr_input._run_cli([command_name]) == 0
    assert events == ["20260408"]


def test_tokenize_text_for_input_preserves_newlines_and_symbols():
    tokens = ehr_input._tokenize_text_for_input("S:\n発熱（37％）")

    assert tokens == [
        {"kind": "ascii", "text": "S:"},
        {"kind": "newline", "text": "\n"},
        {"kind": "japanese", "text": "発熱"},
        {"kind": "ascii", "text": "(37%)"},
    ]


def test_segment_japanese_with_default_vlm_falls_back_to_local(monkeypatch):
    monkeypatch.setattr(
        ehr_input,
        "segment_japanese_text_with_mlx_vlm",
        lambda text: (_ for _ in ()).throw(MlxVlmSegmentationError("server down")),
    )
    monkeypatch.setattr(
        ehr_input,
        "segment_japanese_text_locally",
        lambda text: ("summary", [{"text": "肺炎", "romaji": "haien"}]),
    )

    assert ehr_input._segment_japanese_with_default_vlm("肺炎") == [
        {"text": "肺炎", "romaji": "haien"}
    ]


def test_segment_japanese_with_default_vlm_rebuilds_romaji(monkeypatch):
    monkeypatch.setattr(
        ehr_input,
        "segment_japanese_text_with_mlx_vlm",
        lambda text: (
            '[{"text":"肺炎","romaji":"wrong"}]',
            [{"text": "肺炎", "romaji": "wrong"}],
        ),
    )
    monkeypatch.setattr(ehr_input, "_kanji_to_romaji", lambda text: "haien")

    assert ehr_input._segment_japanese_with_default_vlm("肺炎") == [
        {"text": "肺炎", "romaji": "haien"}
    ]


def test_type_ascii_text_precisely_uses_special_key_commands():
    events = []

    class DummyClient:
        def type_text(self, text):
            events.append(("type", text))
            return True

        def press_key(self, key):
            events.append(("key", key))
            return True

    ehr_input._type_ascii_text_precisely(DummyClient(), "A[1]\n(5%)")

    assert events == [
        ("type", "A"),
        ("key", "lbracket"),
        ("type", "1"),
        ("key", "rbracket"),
        ("key", "enter"),
        ("key", "lparen"),
        ("type", "5"),
        ("key", "percent"),
        ("key", "rparen"),
    ]


def test_parse_ime_candidate_response_reads_json_candidate():
    assert ehr_input._parse_ime_candidate_response('{"candidate":"微熱"}') == "微熱"
    assert ehr_input._parse_ime_candidate_response('{"candidate":null}') is None


def test_should_fallback_to_local_segmentation_for_single_kanji_run():
    assert ehr_input._should_fallback_to_local_segmentation(
        [{"text": "咽"}, {"text": "頭"}, {"text": "痛"}]
    )
    assert ehr_input._should_fallback_to_local_segmentation(
        [{"text": "使"}, {"text": "った"}]
    )
    assert not ehr_input._should_fallback_to_local_segmentation(
        [{"text": "感冒"}, {"text": "症状"}]
    )


def _make_frame(h=200, w=400) -> np.ndarray:
    """Blank BGR frame for testing."""
    return np.zeros((h, w, 3), dtype=np.uint8)



def test_detect_ime_mode_returns_japanese_via_vlm(monkeypatch):
    """detect_ime_mode は VLM が 'あ' を返せば 'japanese' を返す。"""
    client = MagicMock()
    config = MagicMock()
    config.capture_device_index = 0
    config.capture_width = 1920
    config.capture_height = 1080

    frame = _make_frame()
    monkeypatch.setattr(ehr_input, "capture_screen", lambda **kw: frame)
    monkeypatch.setattr(ehr_input, "detect_ime_mode_from_typed_a", lambda f: "japanese")

    assert ehr_input.detect_ime_mode(client, config) == "japanese"
    client.type_text.assert_called_once_with("a")
    client.press_key.assert_any_call("escape")


def test_detect_ime_mode_returns_english_via_vlm(monkeypatch):
    """detect_ime_mode は VLM が 'a' を返せば 'english' を返す。"""
    client = MagicMock()
    config = MagicMock()
    config.capture_device_index = 0
    config.capture_width = 1920
    config.capture_height = 1080

    frame = _make_frame()
    monkeypatch.setattr(ehr_input, "capture_screen", lambda **kw: frame)
    monkeypatch.setattr(ehr_input, "detect_ime_mode_from_typed_a", lambda f: "english")

    assert ehr_input.detect_ime_mode(client, config) == "english"
    client.press_key.assert_called_with("backspace")


def test_detect_ime_mode_returns_none_when_capture_fails(monkeypatch):
    """capture_screen が None を返すと detect_ime_mode は None を返す。"""
    client = MagicMock()
    config = MagicMock()
    config.capture_device_index = 0
    config.capture_width = 1920
    config.capture_height = 1080

    monkeypatch.setattr(ehr_input, "capture_screen", lambda **kw: None)

    assert ehr_input.detect_ime_mode(client, config) is None
