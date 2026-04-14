from pathlib import Path
from unittest.mock import patch, MagicMock

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

    monkeypatch.setattr(ehr_input, "_input_resolved_text", lambda text: events.append(text))

    assert ehr_input._run_cli([str(note)]) == 0
    assert events == ["肺炎に対する治療"]


def test_run_cli_uses_file_contents_for_open_test(monkeypatch, tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("COVID-19の感染を確認した", encoding="utf-8")
    events = []

    monkeypatch.setattr(ehr_input, "open_test_patient_chart", lambda: events.append("open"))
    monkeypatch.setattr(ehr_input, "_input_resolved_text", lambda text: events.append(text))

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


def _mock_match_template(scores: dict):
    """Return a cv2.matchTemplate mock that yields scores by call order."""
    call_iter = iter(scores.values())

    def fake_match(src, tmpl, method):
        score = next(call_iter)
        result = np.full((1, 1), score, dtype=np.float32)
        return result

    return fake_match


def test_detect_ime_mode_returns_japanese_when_hiragana_template_wins():
    frame = _make_frame()
    # 各テンプレートグループ (english×2, hiragana×2) に対して matchTemplate が呼ばれる
    with patch("cv2.imread", return_value=np.zeros((20, 159, 3), dtype=np.uint8)), \
         patch("cv2.matchTemplate", side_effect=[
             np.full((1, 1), 0.5, dtype=np.float32),   # english_ime score
             np.full((1, 1), 0.4, dtype=np.float32),   # english_ime2 score
             np.full((1, 1), 0.9, dtype=np.float32),   # hiragana_ime score
             np.full((1, 1), 0.8, dtype=np.float32),   # hiragana_ime2 score
         ]):
        assert ehr_input.detect_ime_mode(frame) == "japanese"


def test_detect_ime_mode_returns_english_when_english_template_wins():
    frame = _make_frame()
    with patch("cv2.imread", return_value=np.zeros((28, 156, 3), dtype=np.uint8)), \
         patch("cv2.matchTemplate", side_effect=[
             np.full((1, 1), 0.85, dtype=np.float32),  # english_ime score
             np.full((1, 1), 0.80, dtype=np.float32),  # english_ime2 score
             np.full((1, 1), 0.4, dtype=np.float32),   # hiragana_ime score
             np.full((1, 1), 0.3, dtype=np.float32),   # hiragana_ime2 score
         ]):
        assert ehr_input.detect_ime_mode(frame) == "english"


def test_detect_ime_mode_returns_none_when_both_below_threshold():
    frame = _make_frame()
    with patch("cv2.imread", return_value=np.zeros((20, 156, 3), dtype=np.uint8)), \
         patch("cv2.matchTemplate", side_effect=[
             np.full((1, 1), 0.3, dtype=np.float32),
             np.full((1, 1), 0.2, dtype=np.float32),
             np.full((1, 1), 0.2, dtype=np.float32),
             np.full((1, 1), 0.1, dtype=np.float32),
         ]):
        assert ehr_input.detect_ime_mode(frame) is None
