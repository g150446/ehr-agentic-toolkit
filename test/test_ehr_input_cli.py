from contextlib import redirect_stdout
from pathlib import Path
import io
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
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


def test_run_cli_parses_openrouter_and_mactest(monkeypatch):
    configured = {}
    events = []

    monkeypatch.setattr(
        ehr_input,
        "_configure_runtime",
        lambda **kwargs: configured.update(kwargs),
    )
    monkeypatch.setattr(
        ehr_input,
        "_input_resolved_text",
        lambda text, **kwargs: events.append((text, kwargs)),
    )

    assert (
        ehr_input._run_cli(
            ["--mactest", "--openrouter", "qwen/qwen3.5-9b", "--win10", "肺炎"]
        )
        == 0
    )
    assert configured == {"mactest": True, "openrouter_model": "qwen/qwen3.5-9b"}
    assert events == [("肺炎", {"windows_version": "windows10", "clear_field": False})]


def test_configure_runtime_openrouter_updates_segmentation_and_ime(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "token-xyz")

    ehr_input._configure_runtime(mactest=False, openrouter_model="qwen/vision-model")

    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_URL == "https://openrouter.ai/api/v1/chat/completions"
    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_MODEL == "qwen/vision-model"
    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_API_KEY == "token-xyz"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_IME_URL == "https://openrouter.ai/api/v1/chat/completions"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_IME_MODEL == "qwen/vision-model"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_IME_API_KEY == "token-xyz"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_URL == "https://openrouter.ai/api/v1/chat/completions"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_MODEL == "qwen/vision-model"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_API_KEY == "token-xyz"


def test_configure_runtime_without_openrouter_restores_defaults(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "token-xyz")
    ehr_input._configure_runtime(mactest=False, openrouter_model="qwen/vision-model")

    ehr_input._configure_runtime(mactest=False, openrouter_model=None)

    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_URL == ehr_input._DEFAULT_SEGMENTATION_RUNTIME["url"]
    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_MODEL == ehr_input._DEFAULT_SEGMENTATION_RUNTIME["model"]
    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_API_KEY == ehr_input._DEFAULT_SEGMENTATION_RUNTIME["api_key"]
    assert ehr_input.mlx_vlm_ime.MLX_VLM_IME_URL == ehr_input._DEFAULT_IME_RUNTIME["url"]
    assert ehr_input.mlx_vlm_ime.MLX_VLM_IME_MODEL == ehr_input._DEFAULT_IME_RUNTIME["model"]
    assert ehr_input.mlx_vlm_ime.MLX_VLM_IME_API_KEY == ehr_input._DEFAULT_IME_RUNTIME["api_key"]
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_URL == ehr_input._DEFAULT_IME_TEXT_RUNTIME["url"]
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_MODEL == ehr_input._DEFAULT_IME_TEXT_RUNTIME["model"]
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_API_KEY == ehr_input._DEFAULT_IME_TEXT_RUNTIME["api_key"]


def test_run_cli_prioritizes_command_over_same_named_file(monkeypatch, tmp_path):
    command_name = "click history 20260408"
    Path(tmp_path / command_name).write_text("dummy", encoding="utf-8")
    events = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ehr_input, "click_history", lambda date_str: events.append(date_str))

    assert ehr_input._run_cli([command_name]) == 0
    assert events == ["20260408"]


def test_input_resolved_text_bypasses_ime_conversion_for_hiragana(monkeypatch):
    events = []

    monkeypatch.setattr(ehr_input, "type_japanese_sentence", lambda text, **kw: events.append(("sentence", text, kw)))
    monkeypatch.setattr(ehr_input, "type_kanji_via_ime", lambda *args, **kwargs: events.append(("ime", args, kwargs)))

    ehr_input._input_resolved_text("てすと", windows_version="windows7", clear_field=True)

    assert events == [("sentence", "てすと", {"windows_version": "windows7", "clear_field": True})]


def test_input_resolved_text_bypasses_ime_conversion_for_katakana(monkeypatch):
    events = []

    monkeypatch.setattr(ehr_input, "type_japanese_sentence", lambda text, **kw: events.append(("sentence", text, kw)))
    monkeypatch.setattr(ehr_input, "type_kanji_via_ime", lambda *args, **kwargs: events.append(("ime", args, kwargs)))

    ehr_input._input_resolved_text("テスト", windows_version="windows10", clear_field=False)

    assert events == [("sentence", "テスト", {"windows_version": "windows10", "clear_field": False})]


def test_segment_text_for_input_splits_kaboucho_into_stable_units():
    assert ehr_input._segment_text_for_input("過膨張") == [
        {"text": "過", "romaji": "ka"},
        {"text": "膨張", "romaji": "bouchou"},
    ]


def test_segment_text_for_input_uses_correct_abg_override():
    assert ehr_input._segment_text_for_input("動脈血ガス") == [
        {"text": "動脈血", "romaji": "doumyakuketsu"},
        {"text": "ガス", "romaji": "gasu"},
    ]


def test_katakana_to_romaji_replaces_nakaguro_with_slash():
    """・(nakaguro) must be replaced with / for JIS IME; non-ASCII chars sent
    to the ESP32 produce unpredictable HID events (0xE3 = Win key)."""
    from automation.local_segmentation import _katakana_to_romaji
    result = _katakana_to_romaji("ソル・コーテフ")
    assert result == "soru/ko-tefu"
    assert result.isascii(), f"Romaji must be ASCII-only, got {result!r}"


def test_ble_client_type_text_rejects_non_ascii():
    """BLE type_text must reject non-ASCII to prevent HID key injection."""
    from automation.ble_client import BLEClient
    import socket
    client = BLEClient.__new__(BLEClient)
    with pytest.raises(ValueError, match="non-ASCII"):
        client.type_text("soru・ko-tefu")


def test_capture_run_output_tees_stdout_and_stderr(tmp_path):
    stdout = io.StringIO()
    stderr = io.StringIO()
    log_path = tmp_path / "logs" / "0417_0000.txt"

    with ehr_input._capture_run_output(log_path, stdout=stdout, stderr=stderr):
        print("hello stdout")
        print("hello stderr", file=__import__("sys").stderr)

    assert stdout.getvalue() == "hello stdout\n"
    assert stderr.getvalue() == "hello stderr\n"
    assert log_path.read_text(encoding="utf-8") == "hello stdout\nhello stderr\n"


def test_build_run_log_path_uses_mmdd_hhmm_name():
    path = ehr_input._build_run_log_path(ehr_input.datetime(2026, 4, 17, 13, 50, 1))

    assert path == ehr_input._RUN_LOGS_DIR / "0417_1350.txt"


def test_build_run_log_path_adds_numeric_suffix_when_name_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(ehr_input, "_RUN_LOGS_DIR", tmp_path / "logs")
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs" / "0417_1350.txt").write_text("", encoding="utf-8")
    (tmp_path / "logs" / "0417_1350_2.txt").write_text("", encoding="utf-8")

    path = ehr_input._build_run_log_path(ehr_input.datetime(2026, 4, 17, 13, 50, 1))

    assert path == tmp_path / "logs" / "0417_1350_3.txt"


def test_build_run_log_header_records_executable_and_options():
    raw_args = ["--mactest", "--openrouter", "qwen/qwen3.5-9b", "--win10", "--clear", "open test", "肺炎"]
    positional_args, option_summary = ehr_input._parse_cli_options(raw_args)

    header = ehr_input._build_run_log_header(
        "/tmp/automation/ehr_input.py",
        raw_args,
        positional_args,
        option_summary,
    )

    assert "=== ehr_input invocation ===" in header
    assert "executable: ehr_input.py" in header
    assert 'argv: ["/tmp/automation/ehr_input.py", "--mactest", "--openrouter", "qwen/qwen3.5-9b", "--win10", "--clear", "open test", "肺炎"]' in header
    assert 'parsed_options: {"clear_field": true, "mactest": true, "openrouter_model": "qwen/qwen3.5-9b", "win10": true, "windows_version": "windows10"}' in header
    assert 'positional_args: ["open test", "肺炎"]' in header


def test_main_prepends_run_header_to_log(monkeypatch, tmp_path):
    monkeypatch.setattr(ehr_input, "_RUN_LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(ehr_input, "_print_usage", lambda: print("usage called"))
    monkeypatch.setattr(sys, "argv", ["/tmp/automation/ehr_input.py", "--win10", "help"])

    assert ehr_input.main(["--win10", "help"]) == 0

    log_files = sorted((tmp_path / "logs").glob("*.txt"))
    assert len(log_files) == 1

    log_text = log_files[0].read_text(encoding="utf-8")
    assert log_text.startswith("=== ehr_input invocation ===\nexecutable: ehr_input.py\n")
    assert 'parsed_options: {"clear_field": false, "mactest": false, "openrouter_model": null, "win10": true, "windows_version": "windows10"}' in log_text
    assert 'positional_args: ["help"]' in log_text
    assert "usage called\n" in log_text


def test_find_best_candidate_match_skips_romaji_fallback_for_pure_kanji(monkeypatch):
    monkeypatch.setattr(
        ehr_input,
        "_kanji_to_romaji",
        lambda text: {"検査": "kensa", "兼さ": "kensa"}[text],
    )

    assert ehr_input._find_best_candidate_match("検査", [(5, "兼さ")]) is None


def test_find_best_candidate_match_rejects_katakana_when_target_has_kanji(monkeypatch):
    """著明な→チョメイナ bug: romaji fallback must not accept pure katakana
    when the target contains kanji."""
    monkeypatch.setattr(
        ehr_input,
        "_kanji_to_romaji",
        lambda text: {
            "著明な": "chomeina",
            "署名な": "shomeina",
            "署明な": "chomeina",
            "ちょめいな": "chomeina",
            "チョメイナ": "chomeina",
        }[text],
    )
    candidates = [(1, "署名な"), (2, "署明な"), (3, "ちょめいな"), (4, "チョメイナ")]
    result = ehr_input._find_best_candidate_match("著明な", candidates)
    # Should match via visual-confusible fifth pass (署明な), NOT romaji katakana
    assert result == (2, "署明な")


def test_find_best_candidate_match_keeps_romaji_fallback_for_non_kanji_target(monkeypatch):
    monkeypatch.setattr(
        ehr_input,
        "_kanji_to_romaji",
        lambda text: {"あい": "ai", "アイ": "ai"}[text],
    )

    assert ehr_input._find_best_candidate_match("あい", [(3, "アイ")]) == (3, "アイ")


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


def test_segment_japanese_with_default_vlm_uses_vlm_romaji(monkeypatch):
    """VLM が返した romaji をそのまま使う（pykakasi で上書きしない）。"""
    monkeypatch.setattr(
        ehr_input,
        "segment_japanese_text_with_mlx_vlm",
        lambda text: (
            '[{"text":"上気道炎","romaji":"joukidouen"}]',
            [{"text": "上気道炎", "romaji": "joukidouen"}],
        ),
    )

    assert ehr_input._segment_japanese_with_default_vlm("上気道炎") == [
        {"text": "上気道炎", "romaji": "joukidouen"}
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
    monkeypatch.setattr(
        ehr_input,
        "detect_ime_mode_from_typed_a",
        lambda f, pre_frame=None: "japanese",
    )

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
    monkeypatch.setattr(
        ehr_input,
        "detect_ime_mode_from_typed_a",
        lambda f, pre_frame=None: "english",
    )

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


def test_read_popup_candidates_with_fallback_uses_vlm_when_ocr_is_sparse(monkeypatch):
    frame = object()

    monkeypatch.setattr(
        ehr_input,
        "read_popup_candidates_numbered",
        lambda image, debug_name="": [(5, "埜")] if image is frame else [],
    )
    monkeypatch.setattr(
        ehr_input.mlx_vlm_ime,
        "read_popup_candidates_numbered_vlm",
        lambda image, debug_name="": [(1, "野"), (2, "弥")] if image is frame else [],
    )

    assert ehr_input._read_popup_candidates_with_fallback("野", frame) == [
        (1, "野"),
        (2, "弥"),
        (5, "埜"),
    ]


def test_read_helper_popup_candidates_prefers_vlm_for_helper_word(monkeypatch):
    frame = object()

    monkeypatch.setattr(
        ehr_input.mlx_vlm_ime,
        "read_popup_candidates_numbered_vlm",
        lambda image, debug_name="": [(1, "過剰"), (2, "箇条")] if image is frame else [],
    )
    monkeypatch.setattr(
        ehr_input.mlx_vlm_ime,
        "read_popup_candidates_ocr",
        lambda image, debug_name="": [(2, "箇条"), (3, "潟状")] if image is frame else [],
    )

    assert ehr_input._read_helper_popup_candidates("過剰", frame) == [
        (1, "過剰"),
        (2, "箇条"),
        (3, "潟状"),
    ]


def test_try_helper_word_fallback_cycles_until_wrapped_candidate(monkeypatch):
    events = []
    frame = _make_frame()
    highlighted = iter(["箇条", "渦状", "個条", "家常", "力条", "嘉承", "箇多", "嘉場", "過剰"])

    class DummyClient:
        def type_text(self, text):
            events.append(("type", text))
            return True

        def press_key(self, key):
            events.append(("key", key))
            return True

        def send_command(self, command):
            events.append(("command", command))
            return True

    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)

    monkeypatch.setattr(
        ehr_input,
        "suggest_ime_helper_word",
        lambda target: [{"word": "過剰", "backspace_count": 1}],
    )
    monkeypatch.setattr(
        ehr_input,
        "_kanji_to_romaji",
        lambda text: {"過剰": "kajou", "膨張": "bouchou"}[text],
    )
    monkeypatch.setattr(ehr_input, "_cancel_ime_popup_safe", lambda *args, **kwargs: None)
    monkeypatch.setattr(ehr_input, "_clear_pending_ime_composition", lambda *args, **kwargs: None)
    monkeypatch.setattr(ehr_input, "_save_debug_image", lambda *args, **kwargs: None)
    monkeypatch.setattr(ehr_input, "capture_screen", lambda **kwargs: frame)
    monkeypatch.setattr(
        ehr_input,
        "_read_helper_popup_candidates",
        lambda helper_word, image, debug_name="": [
            (2, "箇条"),
            (3, "潟状"),
            (5, "家京"),
            (6, "力条"),
            (8, "力条"),
            (9, "ヶ手"),
        ],
    )
    monkeypatch.setattr(
        ehr_input,
        "read_highlighted_popup_candidate",
        lambda image, debug_name="": next(highlighted),
    )
    monkeypatch.setattr(
        ehr_input,
        "type_kanji_via_ime",
        lambda romaji, target, **kwargs: events.append(("ime", romaji, target, kwargs)),
    )

    assert ehr_input._try_helper_word_fallback(DummyClient(), config, "過膨張", 0.0, "windows10")
    assert events.count(("key", "space")) == 10
    assert ("key", "enter") in events
    assert ("key", "backspace") in events
    assert ("ime", "bouchou", "膨張", {"wait_sec": 0.0, "windows_version": "windows10", "_current_ime_mode": "japanese"}) in events


def test_try_helper_word_fallback_cleans_up_after_backspace(monkeypatch):
    events = []
    frame = _make_frame()

    class DummyClient:
        def type_text(self, text):
            events.append(("type", text))
            return True

        def press_key(self, key):
            events.append(("key", key))
            return True

        def send_command(self, command):
            events.append(("command", command))
            return True

    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)

    monkeypatch.setattr(
        ehr_input,
        "suggest_ime_helper_word",
        lambda target: [{"word": "過剰", "backspace_count": 1}],
    )
    monkeypatch.setattr(
        ehr_input,
        "_kanji_to_romaji",
        lambda text: {"過剰": "kajou", "膨張": "bouchou"}[text],
    )
    monkeypatch.setattr(ehr_input, "_cancel_ime_popup_safe", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ehr_input,
        "_cleanup_after_helper_backspace",
        lambda client, config, helper_word, backspace_count: events.append(
            ("cleanup", helper_word, backspace_count)
        ),
    )
    monkeypatch.setattr(ehr_input, "_save_debug_image", lambda *args, **kwargs: None)
    monkeypatch.setattr(ehr_input, "capture_screen", lambda **kwargs: frame)
    monkeypatch.setattr(
        ehr_input,
        "_read_helper_popup_candidates",
        lambda helper_word, image, debug_name="": [(1, "過剰")],
    )
    monkeypatch.setattr(
        ehr_input,
        "type_kanji_via_ime",
        lambda romaji, target, **kwargs: events.append(("ime", romaji, target, kwargs)),
    )

    assert ehr_input._try_helper_word_fallback(DummyClient(), config, "過膨張", 0.0, "windows10")
    assert ("cleanup", "過剰", 1) in events
    assert events.index(("cleanup", "過剰", 1)) < events.index(
        ("ime", "bouchou", "膨張", {"wait_sec": 0.0, "windows_version": "windows10", "_current_ime_mode": "japanese"})
    )


def test_try_helper_word_fallback_clears_ime_before_helper_lookup(monkeypatch):
    events = []
    frame = _make_frame()

    class DummyClient:
        def type_text(self, text):
            events.append(("type", text))
            return True

        def press_key(self, key):
            events.append(("key", key))
            return True

        def send_command(self, command):
            events.append(("command", command))
            return True

    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)

    monkeypatch.setattr(
        ehr_input,
        "_cancel_ime_popup_safe",
        lambda client, text, wait=0.15, config=None: events.append(("cancel", text)),
    )
    monkeypatch.setattr(
        ehr_input,
        "_clear_pending_ime_composition",
        lambda client, config, max_backspaces: events.append(("clear", max_backspaces)),
    )
    monkeypatch.setattr(
        ehr_input,
        "suggest_ime_helper_word",
        lambda target: events.append(("suggest", target)) or [{"word": "過剰", "backspace_count": 1}],
    )
    monkeypatch.setattr(ehr_input, "_kanji_to_romaji", lambda text: {"過剰": "kajou"}[text])
    monkeypatch.setattr(ehr_input, "_save_debug_image", lambda *args, **kwargs: None)
    monkeypatch.setattr(ehr_input, "capture_screen", lambda **kwargs: frame)
    monkeypatch.setattr(
        ehr_input,
        "_read_helper_popup_candidates",
        lambda helper_word, image, debug_name="": [(1, "過剰")],
    )

    assert ehr_input._try_helper_word_fallback(DummyClient(), config, "過", 0.0, "windows10")
    assert events[:3] == [("cancel", "過"), ("clear", 2), ("suggest", "過")]


def test_fallback_remaining_after_prefix_cancels_and_reinputs(monkeypatch):
    events = []
    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)

    monkeypatch.setattr(
        ehr_input,
        "_cancel_ime_popup_safe",
        lambda client, text, wait=0.15, config=None: events.append(("cancel", text)),
    )
    monkeypatch.setattr(
        ehr_input,
        "_clear_pending_ime_composition",
        lambda client, config, max_backspaces: events.append(("clear", max_backspaces)),
    )
    monkeypatch.setattr(ehr_input, "_kanji_to_romaji", lambda text: {"血": "ketsu"}[text])
    monkeypatch.setattr(
        ehr_input,
        "type_kanji_via_ime",
        lambda romaji, target, **kwargs: events.append(("ime", romaji, target, kwargs)),
    )

    ehr_input._fallback_remaining_after_prefix(
        object(),
        config,
        remaining_target="血",
        wait_sec=0.0,
        windows_version="windows7",
    )

    assert events[0] == ("cancel", "血")
    assert events[1][0] == "clear"
    assert events[2] == ("ime", "ketsu", "血", {"wait_sec": 0.0, "windows_version": "windows7", "_current_ime_mode": "japanese"})


def test_cancel_ime_popup_safe_uses_fixed_bs_then_vlm_guard_when_config_available(monkeypatch):
    events = []
    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

    monkeypatch.setattr(ehr_input, "_text_to_hiragana_len", lambda text: 2)
    monkeypatch.setattr(
        ehr_input,
        "_clear_pending_ime_composition",
        lambda client, config, max_backspaces: events.append(("guarded-clear", max_backspaces)),
    )

    ehr_input._cancel_ime_popup_safe(DummyClient(), "過", config=config)

    # Esc (popup→inline) + F6 (inline→hiragana) + BS×hira_len + VLM guard
    assert events == [
        ("key", "escape"),
        ("key", "f6"),
        ("key", "backspace"),
        ("key", "backspace"),
        ("guarded-clear", 2),
    ]


def test_cancel_ime_popup_safe_no_config_uses_f6_then_bs(monkeypatch):
    """config=None の場合も F6 + BS×hira_len を送る（VLM ガードなし）。"""
    events = []

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

    monkeypatch.setattr(ehr_input, "_text_to_hiragana_len", lambda text: 1)

    ehr_input._cancel_ime_popup_safe(DummyClient(), "過", config=None)

    assert events == [
        ("key", "escape"),
        ("key", "f6"),
        ("key", "backspace"),
    ]


def test_clear_pending_ime_composition_skips_trailing_esc_when_no_composition(monkeypatch):
    """VLM が組成なしと判定した場合、trailing Esc を送らない（Esc が誤確定を起こすため）。"""
    events = []
    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

    monkeypatch.setattr(ehr_input, "_capture_frame", lambda config: _make_frame())
    monkeypatch.setattr(ehr_input, "_has_ime_composition", lambda frame: False)

    ehr_input._clear_pending_ime_composition(DummyClient(), config, max_backspaces=4)

    # No backspace sent, so no trailing escape either
    assert events == []


def test_clear_pending_ime_composition_sends_trailing_esc_after_clearing(monkeypatch):
    """VLM が組成を検出して BS を送った場合のみ、trailing Esc を送る。"""
    events = []
    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)
    composition_present = iter([True, False])

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

    monkeypatch.setattr(ehr_input, "_capture_frame", lambda config: _make_frame())
    monkeypatch.setattr(ehr_input, "_has_ime_composition", lambda frame: next(composition_present))

    ehr_input._clear_pending_ime_composition(DummyClient(), config, max_backspaces=4)

    assert events == [("key", "backspace"), ("key", "escape")]
    events = []

    class DummyClient:
        def type_text(self, text):
            events.append(("type", text))
            return True

        def press_key(self, key):
            events.append(("key", key))
            return True

    monkeypatch.setattr(ehr_input, "load_config", lambda skip_password=True: SimpleNamespace())
    monkeypatch.setattr(ehr_input, "_wait_for_ble_connected", lambda: DummyClient())
    monkeypatch.setattr(ehr_input, "detect_ime_mode", lambda *args, **kwargs: "japanese")
    monkeypatch.setattr(ehr_input, "ensure_ime_mode", lambda target, client, current: target)
    monkeypatch.setattr(
        ehr_input,
        "_iter_segments_for_input",
        lambda text: iter([{"text": "、", "romaji": ","}]),
    )

    ehr_input.type_japanese_sentence("、", windows_version="windows7")

    assert events == [("type", ","), ("key", "enter")]


def test_segment_japanese_with_openrouter_uses_runtime_aware_logs(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "token-xyz")
    monkeypatch.setattr(
        ehr_input,
        "segment_japanese_text_with_mlx_vlm",
        lambda text: ('[{"text":"肺炎","romaji":"haien"}]', [{"text": "肺炎", "romaji": "haien"}]),
    )
    monkeypatch.setattr(ehr_input, "_kanji_to_romaji", lambda text: "haien")
    stdout = io.StringIO()

    ehr_input._configure_runtime(mactest=False, openrouter_model="google/gemma-4-26b-a4b-it")
    try:
        with redirect_stdout(stdout):
            assert ehr_input._segment_japanese_with_default_vlm("肺炎") == [
                {"text": "肺炎", "romaji": "haien"}
            ]
    finally:
        ehr_input._configure_runtime(mactest=False, openrouter_model=None)

    output = stdout.getvalue()
    assert "OpenRouter(google/gemma-4-26b-a4b-it)分割結果" in output
    assert "Qwen分割結果" not in output


def test_wait_for_ble_connected_returns_local_client_in_mactest(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(ehr_input._RUNTIME_OPTIONS, "mactest", True)
    monkeypatch.setattr(ehr_input._RUNTIME_OPTIONS, "local_client", sentinel)

    assert ehr_input._wait_for_ble_connected() is sentinel


def test_capture_screen_accepts_flush_duration_in_mactest(monkeypatch):
    fake_rgb = np.zeros((8, 12, 3), dtype=np.uint8)

    class FakePyAutoGUI:
        def screenshot(self):
            return fake_rgb

    fake_client = type("FakeClient", (), {"_pyautogui": FakePyAutoGUI()})()
    monkeypatch.setattr(ehr_input._RUNTIME_OPTIONS, "mactest", True)
    monkeypatch.setattr(ehr_input._RUNTIME_OPTIONS, "local_client", fake_client)

    frame = ehr_input.capture_screen(
        device_index=0,
        width=1920,
        height=1080,
        flush_duration=0.5,
    )

    assert frame.shape == (8, 12, 3)


def test_type_kanji_via_ime_aborts_before_typing_when_capture_unavailable(monkeypatch):
    events = []
    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)

    class DummyClient:
        def type_text(self, text):
            events.append(("type", text))
            return True

        def press_key(self, key):
            events.append(("key", key))
            return True

    monkeypatch.setattr(ehr_input, "load_config", lambda skip_password=True: config)
    monkeypatch.setattr(ehr_input, "_wait_for_ble_connected", lambda timeout=70.0: DummyClient())
    monkeypatch.setattr(ehr_input, "capture_screen", lambda **kwargs: None)

    with pytest.raises(RuntimeError, match="HDMIキャプチャデバイスからフレームを取得できませんでした"):
        ehr_input.type_kanji_via_ime("choushin", "聴診", _current_ime_mode="japanese")

    assert events == []
