from contextlib import redirect_stdout
from pathlib import Path
import io
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import automation.ehr_input as ehr_input
import automation.mlx_vlm_ime as mlx_vlm_ime
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


def test_find_patient_search_tab_accepts_fuzzy_ocr_read():
    tab_bbox = [[14, 100], [80, 100], [80, 124], [14, 124]]
    other_bbox = [[204, 102], [296, 102], [296, 126], [204, 126]]

    match = ehr_input._find_patient_search_tab(
        [
            (other_bbox, "受付患若一覧", 0.27),
            (tab_bbox, "愚着検索", 0.47),
        ]
    )

    assert match == (47, 112, "愚着検索", 0.47)


def test_open_test_patient_chart_clicks_fuzzy_patient_search_tab(monkeypatch):
    events = []
    frame = _make_frame(1080, 1920)
    config = SimpleNamespace(
        capture_device_index=0,
        capture_width=1920,
        capture_height=1080,
    )

    class DummyClient:
        def switch_to_mouse_mode(self):
            events.append(("mode", "mouse"))
            return True

        def move_mouse_to_position(self, x, y):
            events.append(("moveto", x, y))
            return True

        def click(self):
            events.append(("click",))
            return True

        def switch_to_keyboard_mode(self):
            events.append(("mode", "keyboard"))
            return True

        def press_key(self, key):
            events.append(("key", key))
            return True

    monkeypatch.setattr(ehr_input, "load_config", lambda skip_password=True: config)
    monkeypatch.setattr(ehr_input, "capture_screen", lambda **kwargs: frame)
    monkeypatch.setattr(
        ehr_input,
        "_request_ocr_results",
        lambda frame, config: [
            ([[14, 100], [80, 100], [80, 124], [14, 124]], "愚着検索", 0.47),
            ([[204, 102], [296, 102], [296, 126], [204, 126]], "受付患若一覧", 0.27),
        ],
    )
    monkeypatch.setattr(ehr_input, "_wait_for_ble_connected", lambda: DummyClient())
    monkeypatch.setattr(
        ehr_input,
        "input_text_to_field",
        lambda **kwargs: events.append(("input", kwargs)),
    )
    monkeypatch.setattr(ehr_input.time, "sleep", lambda *_: None)

    ehr_input.open_test_patient_chart()

    assert ("moveto", 47, 112) in events
    assert ("click",) in events
    assert ("input", {"input_text": "tesuto", "label": "フリガナ"}) in events
    assert events.count(("key", "enter")) == 2


def test_run_cli_parses_openrouter(monkeypatch):
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
            ["--openrouter", "qwen/qwen3.5-9b", "肺炎"]
        )
        == 0
    )
    assert configured == {
        "fireworks_model": None,
        "google_ai_studio": False,
        "novita_model": None,
        "openrouter_model": "qwen/qwen3.5-9b",
    }
    assert events == [("肺炎", {"clear_field": False})]


def test_run_cli_parses_novita_default_model(monkeypatch):
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

    assert ehr_input._run_cli(["--novita", "肺炎"]) == 0
    assert configured == {
        "fireworks_model": None,
        "google_ai_studio": False,
        "novita_model": "google/gemma-4-31b-it",
        "openrouter_model": None,
    }
    assert events == [("肺炎", {"clear_field": False})]


def test_run_cli_parses_novita_custom_model(monkeypatch):
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

    assert ehr_input._run_cli(["--novita", "deepseek/deepseek-vl2", "肺炎"]) == 0
    assert configured == {
        "fireworks_model": None,
        "google_ai_studio": False,
        "novita_model": "deepseek/deepseek-vl2",
        "openrouter_model": None,
    }
    assert events == [("肺炎", {"clear_field": False})]


def test_run_cli_parses_dual_openrouter_and_novita_default_model(monkeypatch):
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

    assert ehr_input._run_cli(["--openrouter", "--novita", "肺炎"]) == 0
    assert configured == {
        "fireworks_model": None,
        "google_ai_studio": False,
        "novita_model": "google/gemma-4-31b-it",
        "openrouter_model": "google/gemma-4-31b-it",
    }
    assert events == [("肺炎", {"clear_field": False})]


def test_run_cli_parses_fireworks(monkeypatch):
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

    assert ehr_input._run_cli(["--fireworks", "accounts/fireworks/models/gemma-4-26b-a4b-it", "肺炎"]) == 0
    assert configured == {
        "fireworks_model": "accounts/fireworks/models/gemma-4-26b-a4b-it",
        "google_ai_studio": False,
        "novita_model": None,
        "openrouter_model": None,
    }
    assert events == [("肺炎", {"clear_field": False})]


def test_run_cli_parses_google_ai_studio(monkeypatch):
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

    assert ehr_input._run_cli(["--google-ai-studio", "肺炎"]) == 0
    assert configured == {
        "fireworks_model": None,
        "google_ai_studio": True,
        "novita_model": None,
        "openrouter_model": None,
    }
    assert events == [("肺炎", {"clear_field": False})]


def test_configure_runtime_openrouter_updates_segmentation_and_ime(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "token-xyz")

    ehr_input._configure_runtime(openrouter_model="qwen/vision-model")

    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_URL == "https://openrouter.ai/api/v1/chat/completions"
    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_MODEL == "qwen/vision-model"
    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_API_KEY == "token-xyz"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_IME_URL == "https://openrouter.ai/api/v1/chat/completions"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_IME_MODEL == "qwen/vision-model"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_IME_API_KEY == "token-xyz"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_URL == "https://openrouter.ai/api/v1/chat/completions"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_MODEL == "qwen/vision-model"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_API_KEY == "token-xyz"


def test_configure_runtime_novita_updates_segmentation_and_ime(monkeypatch):
    monkeypatch.setenv("NOVITA_API_KEY", "token-novita")

    ehr_input._configure_runtime(novita_model="google/gemma-4-31b-it")

    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_URL == "https://api.novita.ai/openai/chat/completions"
    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_MODEL == "google/gemma-4-31b-it"
    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_API_KEY == "token-novita"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_IME_URL == "https://api.novita.ai/openai/chat/completions"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_IME_MODEL == "google/gemma-4-31b-it"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_IME_API_KEY == "token-novita"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_URL == "https://api.novita.ai/openai/chat/completions"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_MODEL == "google/gemma-4-31b-it"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_API_KEY == "token-novita"


def test_configure_runtime_dual_provider_starts_with_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "token-openrouter")
    monkeypatch.setenv("NOVITA_API_KEY", "token-novita")

    ehr_input._configure_runtime(
        openrouter_model="google/gemma-4-31b-it",
        novita_model="google/gemma-4-31b-it",
    )

    assert ehr_input._RUNTIME_OPTIONS.dual_provider_model == "google/gemma-4-31b-it"
    assert ehr_input._RUNTIME_OPTIONS.dual_provider_turn == 0
    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_URL == "https://openrouter.ai/api/v1/chat/completions"
    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_MODEL == "google/gemma-4-31b-it"
    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_API_KEY == "token-openrouter"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_URL == "https://openrouter.ai/api/v1/chat/completions"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_MODEL == "google/gemma-4-31b-it"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_API_KEY == "token-openrouter"


def test_configure_runtime_google_ai_studio_updates_segmentation_and_ime(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "token-gemini")

    ehr_input._configure_runtime(google_ai_studio=True)

    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_URL == "https://generativelanguage.googleapis.com/v1beta"
    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_MODEL == "gemma-4-26b-a4b-it"
    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_API_KEY == "token-gemini"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_IME_URL == "https://generativelanguage.googleapis.com/v1beta"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_IME_MODEL == "gemma-4-26b-a4b-it"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_IME_API_KEY == "token-gemini"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_URL == "https://generativelanguage.googleapis.com/v1beta"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_MODEL == "gemma-4-26b-a4b-it"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_API_KEY == "token-gemini"


def test_configure_runtime_fireworks_updates_segmentation_and_ime(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "token-fireworks")

    ehr_input._configure_runtime(fireworks_model="accounts/fireworks/models/gemma-4-26b-a4b-it")

    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_URL == "https://api.fireworks.ai/inference/v1/chat/completions"
    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_MODEL == "accounts/fireworks/models/gemma-4-26b-a4b-it"
    assert ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_API_KEY == "token-fireworks"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_IME_URL == "https://api.fireworks.ai/inference/v1/chat/completions"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_IME_MODEL == "accounts/fireworks/models/gemma-4-26b-a4b-it"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_IME_API_KEY == "token-fireworks"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_URL == "https://api.fireworks.ai/inference/v1/chat/completions"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_MODEL == "accounts/fireworks/models/gemma-4-26b-a4b-it"
    assert ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_API_KEY == "token-fireworks"


def test_configure_runtime_without_external_provider_restores_defaults(monkeypatch):
    monkeypatch.setenv("NOVITA_API_KEY", "token-novita")
    ehr_input._configure_runtime(novita_model="google/gemma-4-31b-it")

    ehr_input._configure_runtime(openrouter_model=None, novita_model=None, google_ai_studio=False)

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

    ehr_input._input_resolved_text("てすと", clear_field=True)

    assert events == [("sentence", "てすと", {"clear_field": True})]


def test_input_resolved_text_bypasses_ime_conversion_for_katakana(monkeypatch):
    events = []

    monkeypatch.setattr(ehr_input, "type_japanese_sentence", lambda text, **kw: events.append(("sentence", text, kw)))
    monkeypatch.setattr(ehr_input, "type_kanji_via_ime", lambda *args, **kwargs: events.append(("ime", args, kwargs)))

    ehr_input._input_resolved_text("テスト", clear_field=False)

    assert events == [("sentence", "テスト", {"clear_field": False})]


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


def test_normalize_text_for_typing_replaces_celsius_symbol_with_degree_kanji():
    assert ehr_input._normalize_text_for_typing("BT 36.6℃") == "BT 36.6度"


def test_normalize_text_for_typing_converts_other_fullwidth_symbols_to_ascii():
    assert ehr_input._normalize_text_for_typing("（％）：［A］【B】") == "(%):[A][B]"


def test_tokenize_text_for_input_splits_normalized_celsius_out_of_ascii_vitals():
    assert ehr_input._tokenize_text_for_input("BT 36.6℃, BP 122/78 mmHg") == [
        {"kind": "ascii", "text": "BT 36.6"},
        {"kind": "japanese", "text": "度"},
        {"kind": "ascii", "text": ", BP 122/78 mmHg"},
    ]


def test_segment_japanese_with_default_vlm_splits_mixed_katakana_and_kanji(monkeypatch):
    monkeypatch.setattr(
        ehr_input,
        "segment_japanese_text_with_mlx_vlm",
        lambda text: ('[{"text":"アレルギー性","romaji":"arerugi-sei"}]', [{"text": "アレルギー性", "romaji": "arerugi-sei"}]),
    )

    assert ehr_input._segment_japanese_with_default_vlm("アレルギー性") == [
        {"text": "アレルギー", "romaji": "arerugi-"},
        {"text": "性", "romaji": "sei"},
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


def test_build_repo_venv_reexec_command_for_ehr_input(monkeypatch, tmp_path):
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(ehr_input, "_repo_venv_python", lambda: venv_python)
    monkeypatch.setattr(ehr_input.sys, "argv", ["/tmp/automation/ehr_input.py", "--novita", "test"])
    monkeypatch.setattr(ehr_input.sys, "executable", "/usr/bin/python3")

    assert ehr_input._build_repo_venv_reexec_command("cv2") == [
        str(venv_python),
        "-m",
        "automation.ehr_input",
        "--novita",
        "test",
    ]


def test_build_repo_venv_reexec_command_skips_non_ehr_input_invocation(monkeypatch, tmp_path):
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(ehr_input, "_repo_venv_python", lambda: venv_python)
    monkeypatch.setattr(ehr_input.sys, "argv", ["/tmp/pytest", "-q"])
    monkeypatch.setattr(ehr_input.sys, "executable", "/usr/bin/python3")

    assert ehr_input._build_repo_venv_reexec_command("cv2") is None


def test_build_run_log_header_records_executable_and_options():
    raw_args = ["--openrouter", "qwen/qwen3.5-9b", "--clear", "open test", "肺炎"]
    positional_args, option_summary = ehr_input._parse_cli_options(raw_args)

    header = ehr_input._build_run_log_header(
        "/tmp/automation/ehr_input.py",
        raw_args,
        positional_args,
        option_summary,
    )

    assert "=== ehr_input invocation ===" in header
    assert "executable: ehr_input.py" in header
    assert 'argv: ["/tmp/automation/ehr_input.py", "--openrouter", "qwen/qwen3.5-9b", "--clear", "open test", "肺炎"]' in header
    assert 'parsed_options: {"clear_field": true, "dual_provider_mode": false, "fireworks_model": null, "google_ai_studio": false, "novita_model": null, "openrouter_model": "qwen/qwen3.5-9b"}' in header
    assert 'positional_args: ["open test", "肺炎"]' in header


def test_main_prepends_run_header_to_log(monkeypatch, tmp_path):
    monkeypatch.setattr(ehr_input, "_RUN_LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(ehr_input, "_print_usage", lambda: print("usage called"))
    monkeypatch.setattr(sys, "argv", ["/tmp/automation/ehr_input.py", "help"])

    assert ehr_input.main(["help"]) == 0

    log_files = sorted((tmp_path / "logs").glob("*.txt"))
    assert len(log_files) == 1

    log_text = log_files[0].read_text(encoding="utf-8")
    assert log_text.startswith("=== ehr_input invocation ===\nexecutable: ehr_input.py\n")
    assert 'parsed_options: {"clear_field": false, "dual_provider_mode": false, "fireworks_model": null, "google_ai_studio": false, "novita_model": null, "openrouter_model": null}' in log_text
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


def test_tokenize_text_for_input_separates_japanese_brackets_from_ascii_brackets():
    tokens = ehr_input._tokenize_text_for_input("[S]「過」")

    assert tokens == [
        {"kind": "ascii", "text": "[S]"},
        {"kind": "jp_bracket", "text": "「"},
        {"kind": "japanese", "text": "過"},
        {"kind": "jp_bracket", "text": "」"},
    ]


def test_tokenize_text_for_input_isolates_japanese_comma_between_words():
    tokens = ehr_input._tokenize_text_for_input("にて、強制")

    assert tokens == [
        {"kind": "japanese", "text": "にて"},
        {"kind": "jp_punct", "text": "、"},
        {"kind": "japanese", "text": "強制"},
    ]


def test_tokenize_text_for_input_isolates_japanese_period_between_words():
    tokens = ehr_input._tokenize_text_for_input("にて。強制")

    assert tokens == [
        {"kind": "japanese", "text": "にて"},
        {"kind": "jp_punct", "text": "。"},
        {"kind": "japanese", "text": "強制"},
    ]


def test_tokenize_text_for_input_isolates_japanese_middle_dot_between_words():
    tokens = ehr_input._tokenize_text_for_input("ソル・コーテフ")

    assert tokens == [
        {"kind": "japanese", "text": "ソル"},
        {"kind": "jp_punct", "text": "・"},
        {"kind": "japanese", "text": "コーテフ"},
    ]


def test_tokenize_text_for_input_keeps_long_vowel_mark_inside_katakana_run():
    tokens = ehr_input._tokenize_text_for_input("コーテフ")

    assert tokens == [
        {"kind": "japanese", "text": "コーテフ"},
    ]


def test_tokenize_text_for_input_keeps_long_vowel_mark_inside_hiragana_run():
    tokens = ehr_input._tokenize_text_for_input("えーと")

    assert tokens == [
        {"kind": "japanese", "text": "えーと"},
    ]


def test_tokenize_text_for_input_still_isolates_standalone_long_vowel_mark():
    tokens = ehr_input._tokenize_text_for_input("ー強制")

    assert tokens == [
        {"kind": "jp_punct", "text": "ー"},
        {"kind": "japanese", "text": "強制"},
    ]


def test_tokenize_text_for_input_isolates_wave_dash_between_words():
    tokens = ehr_input._tokenize_text_for_input("発熱〜咳嗽")

    assert tokens == [
        {"kind": "japanese", "text": "発熱"},
        {"kind": "jp_punct", "text": "〜"},
        {"kind": "japanese", "text": "咳嗽"},
    ]


def test_update_helper_anchor_text_appends_adjacent_ascii_suffix():
    assert ehr_input._update_helper_anchor_text("症状", "(") == "症状("
    assert ehr_input._update_helper_anchor_text("症状(", "A1") == "症状(A1"


def test_update_helper_anchor_text_resets_on_new_japanese_segment():
    assert ehr_input._update_helper_anchor_text("症状(", "咽頭痛") == "咽頭痛"


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


def test_segment_japanese_with_default_vlm_keeps_cutlet_aligned_romaji(monkeypatch):
    """cutlet と一致する VLM romaji はそのまま使う。"""
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


def test_segment_japanese_with_default_vlm_ignores_vlm_romaji(monkeypatch):
    monkeypatch.setattr(
        ehr_input,
        "segment_japanese_text_with_mlx_vlm",
        lambda text: (
            '[{"text":"日前","romaji":"zenjitsu"}]',
            [{"text": "日前", "romaji": "zenjitsu"}],
        ),
    )

    assert ehr_input._segment_japanese_with_default_vlm("日前") == [
        {"text": "日前", "romaji": "nichimae"}
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


def _make_patient_record_frame() -> np.ndarray:
    frame = np.full((240, 600, 3), 255, dtype=np.uint8)
    for x in (100, 200, 300, 400):
        frame[:, x - 1:x + 2] = 180
    return frame



def test_detect_ime_mode_returns_japanese_via_vlm_without_backspace_when_escape_clears_composition(monkeypatch):
    """detect_ime_mode は Escape 後に組成が消えていれば Backspace を送らない。"""
    client = MagicMock()
    config = MagicMock()
    config.capture_device_index = 0
    config.capture_width = 1920
    config.capture_height = 1080

    frame = _make_frame()
    frames = iter([frame, frame, frame])
    monkeypatch.setattr(ehr_input.time, "sleep", lambda _: None)
    monkeypatch.setattr(ehr_input, "capture_screen", lambda **kw: next(frames))
    monkeypatch.setattr(
        ehr_input,
        "detect_ime_mode_from_typed_a",
        lambda f, pre_frame=None: "japanese",
    )
    monkeypatch.setattr(ehr_input, "_has_ime_composition", lambda _: False)

    assert ehr_input.detect_ime_mode(client, config) == "japanese"
    client.type_text.assert_called_once_with("a")
    assert [call.args[0] for call in client.press_key.call_args_list] == ["escape"]


def test_detect_ime_mode_returns_japanese_via_vlm_with_backspace_when_composition_remains(monkeypatch):
    client = MagicMock()
    config = MagicMock()
    config.capture_device_index = 0
    config.capture_width = 1920
    config.capture_height = 1080

    frame = _make_frame()
    frames = iter([frame, frame, frame])
    monkeypatch.setattr(ehr_input.time, "sleep", lambda _: None)
    monkeypatch.setattr(ehr_input, "capture_screen", lambda **kw: next(frames))
    monkeypatch.setattr(
        ehr_input,
        "detect_ime_mode_from_typed_a",
        lambda f, pre_frame=None: "japanese",
    )
    monkeypatch.setattr(ehr_input, "_has_ime_composition", lambda _: True)

    assert ehr_input.detect_ime_mode(client, config) == "japanese"
    assert [call.args[0] for call in client.press_key.call_args_list] == ["escape", "backspace"]


def test_detect_ime_mode_returns_english_via_vlm(monkeypatch):
    """detect_ime_mode は VLM が 'a' を返せば 'english' を返す。"""
    client = MagicMock()
    config = MagicMock()
    config.capture_device_index = 0
    config.capture_width = 1920
    config.capture_height = 1080

    frame = _make_frame()
    frames = iter([frame, frame])
    monkeypatch.setattr(ehr_input.time, "sleep", lambda _: None)
    monkeypatch.setattr(ehr_input, "capture_screen", lambda **kw: next(frames))
    monkeypatch.setattr(
        ehr_input,
        "detect_ime_mode_from_typed_a",
        lambda f, pre_frame=None: "english",
    )

    assert ehr_input.detect_ime_mode(client, config) == "english"
    assert [call.args[0] for call in client.press_key.call_args_list] == ["backspace"]


def test_detect_ime_mode_returns_none_when_capture_fails(monkeypatch):
    """capture_screen が None を返すと detect_ime_mode は None を返す。"""
    client = MagicMock()
    config = MagicMock()
    config.capture_device_index = 0
    config.capture_width = 1920
    config.capture_height = 1080

    monkeypatch.setattr(ehr_input.time, "sleep", lambda _: None)
    monkeypatch.setattr(ehr_input, "capture_screen", lambda **kw: None)

    assert ehr_input.detect_ime_mode(client, config) is None
    assert [call.args[0] for call in client.press_key.call_args_list] == ["escape"]


def test_detect_ime_mode_returns_none_without_backspace_when_cleanup_finds_no_composition(monkeypatch):
    client = MagicMock()
    config = MagicMock()
    config.capture_device_index = 0
    config.capture_width = 1920
    config.capture_height = 1080

    frame = _make_frame()
    frames = iter([frame, frame, frame])
    monkeypatch.setattr(ehr_input.time, "sleep", lambda _: None)
    monkeypatch.setattr(ehr_input, "capture_screen", lambda **kw: next(frames))
    monkeypatch.setattr(
        ehr_input,
        "detect_ime_mode_from_typed_a",
        lambda f, pre_frame=None: None,
    )

    assert ehr_input.detect_ime_mode(client, config) is None
    assert [call.args[0] for call in client.press_key.call_args_list] == ["escape"]


def test_detect_ime_mode_returns_none_with_backspace_when_probe_text_still_differs(monkeypatch):
    client = MagicMock()
    config = MagicMock()
    config.capture_device_index = 0
    config.capture_width = 1920
    config.capture_height = 1080

    pre_frame = _make_frame()
    post_frame = pre_frame.copy()
    cleanup_frame = pre_frame.copy()
    cleanup_frame[20:26, 20:26] = 255
    frames = iter([pre_frame, post_frame, cleanup_frame])
    monkeypatch.setattr(ehr_input.time, "sleep", lambda _: None)
    monkeypatch.setattr(ehr_input, "capture_screen", lambda **kw: next(frames))
    monkeypatch.setattr(
        ehr_input,
        "detect_ime_mode_from_typed_a",
        lambda f, pre_frame=None: None,
    )

    assert ehr_input.detect_ime_mode(client, config) is None
    assert [call.args[0] for call in client.press_key.call_args_list] == ["escape", "backspace"]


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
    monkeypatch.setattr(
        ehr_input,
        "_reset_ime_before_helper_lookup",
        lambda client, config, target_kanji, left_context="", anchor_text="", baseline_state=None, max_escape_count=4, wait=0.5: True,
    )
    monkeypatch.setattr(
        ehr_input,
        "_capture_helper_reset_baseline",
        lambda config, anchor_text="", debug_name="helper_reset_baseline": {"final_line": "過", "char_count": 1, "anchor_present": True},
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

    assert ehr_input._try_helper_word_fallback(DummyClient(), config, "過膨張", 0.0)
    assert events.count(("key", "space")) == 10
    assert ("key", "enter") in events
    assert ("key", "backspace") in events
    assert (
        "ime",
        "bouchou",
        "膨張",
        {
            "wait_sec": 0.0,
            "_current_ime_mode": "japanese",
            "_typed_prefix_context": "過",
            "_helper_anchor_text": "過",
            "_helper_reset_baseline": {"final_line": "過", "char_count": 1, "anchor_present": True},
        },
    ) in events


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
    monkeypatch.setattr(
        ehr_input,
        "_reset_ime_before_helper_lookup",
        lambda client, config, target_kanji, left_context="", anchor_text="", baseline_state=None, max_escape_count=4, wait=0.5: True,
    )
    monkeypatch.setattr(
        ehr_input,
        "_capture_helper_reset_baseline",
        lambda config, anchor_text="", debug_name="helper_reset_baseline": {"final_line": "過", "char_count": 1, "anchor_present": True},
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

    assert ehr_input._try_helper_word_fallback(DummyClient(), config, "過膨張", 0.0)
    assert ("cleanup", "過剰", 1) in events
    assert events.index(("cleanup", "過剰", 1)) < events.index(
        (
            "ime",
            "bouchou",
            "膨張",
            {
                "wait_sec": 0.0,
                "_current_ime_mode": "japanese",
                "_typed_prefix_context": "過",
                "_helper_anchor_text": "過",
                "_helper_reset_baseline": {"final_line": "過", "char_count": 1, "anchor_present": True},
            },
        )
    )


def test_try_helper_word_fallback_resets_ime_with_vlm_checked_escape_loop_before_helper_lookup(monkeypatch):
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
        "_reset_ime_before_helper_lookup",
        lambda client, config, target_kanji, left_context="", anchor_text="", baseline_state=None, max_escape_count=4, wait=0.5: (
            events.append(("reset", target_kanji, left_context, anchor_text, baseline_state, max_escape_count, wait)) or True
        ),
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

    assert ehr_input._try_helper_word_fallback(DummyClient(), config, "過", 0.0, anchor_text="咽頭")
    assert events[:2] == [("reset", "過", "", "咽頭", None, 4, 0.5), ("suggest", "過")]


def test_try_helper_word_fallback_stops_when_adaptive_reset_is_not_safe(monkeypatch):
    monkeypatch.setattr(
        ehr_input,
        "_reset_ime_before_helper_lookup",
        lambda client, config, target_kanji, left_context="", anchor_text="", baseline_state=None, max_escape_count=4, wait=0.5: False,
    )
    monkeypatch.setattr(
        ehr_input,
        "suggest_ime_helper_word",
        lambda target: (_ for _ in ()).throw(AssertionError("should not query helper words")),
    )

    assert not ehr_input._try_helper_word_fallback(object(), object(), "咽頭痛", 0.0, left_context="昨日から")


def test_fallback_remaining_after_prefix_cancels_and_reinputs(monkeypatch):
    events = []
    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)

    monkeypatch.setattr(
        ehr_input,
        "_cancel_ime_popup_safe",
        lambda client, text, wait=0.15, config=None, romaji="": events.append(("cancel", text)),
    )
    monkeypatch.setattr(
        ehr_input,
        "_clear_pending_ime_composition",
        lambda client, config, max_backspaces: events.append(("clear", max_backspaces)),
    )
    monkeypatch.setattr(ehr_input, "_kanji_to_romaji", lambda text: {"血": "ketsu"}[text])
    monkeypatch.setattr(
        ehr_input,
        "_capture_helper_reset_baseline",
        lambda config, anchor_text="", debug_name="helper_reset_baseline": {"final_line": "咽頭", "char_count": 2, "anchor_present": True},
    )
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
        typed_prefix_context="咽頭",
        helper_anchor_text="咽頭",
    )

    assert events[0] == ("cancel", "血")
    assert events[1][0] == "clear"
    assert events[2] == (
        "ime",
        "ketsu",
        "血",
        {
            "wait_sec": 0.0,
            "_current_ime_mode": "japanese",
            "_typed_prefix_context": "咽頭",
            "_helper_anchor_text": "咽頭",
            "_helper_reset_baseline": {"final_line": "咽頭", "char_count": 2, "anchor_present": True},
        },
    )


def test_cancel_ime_popup_safe_vlm_guided_stops_when_composition_cleared(monkeypatch):
    """VLM-guided path: conservative floor + Phase 2 VLM check stops when no composition."""
    events = []
    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)
    import numpy as np

    def _make_frame():
        return np.zeros((100, 200, 3), dtype=np.uint8)

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

    # VLM: initial=True(組成あり), Phase 2=False(組成なし→追加BSなし)
    composition_iter = iter([True, False])
    monkeypatch.setattr(ehr_input, "_text_to_hiragana_len", lambda text: 4)
    monkeypatch.setattr(ehr_input, "_capture_frame", lambda config: _make_frame())
    monkeypatch.setattr(ehr_input, "_has_ime_composition", lambda frame: next(composition_iter))

    ehr_input._cancel_ime_popup_safe(DummyClient(), "呼気性", config=config)

    # Esc + F6 + conservative floor BS×3 + Phase 2 VLM says gone → no extra BS
    assert events == [
        ("key", "escape"),
        ("key", "f6"),
        ("key", "backspace"),
        ("key", "backspace"),
        ("key", "backspace"),
    ]


def test_cancel_ime_popup_safe_vlm_guided_extra_bs_when_remaining(monkeypatch):
    """VLM-guided path: Phase 2 VLM detects remaining composition → 1 extra BS."""
    events = []
    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)
    import numpy as np

    def _make_frame():
        return np.zeros((100, 200, 3), dtype=np.uint8)

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

    # VLM: initial=True, Phase 2=True(残存あり→追加BS×1)
    composition_iter = iter([True, True])
    monkeypatch.setattr(ehr_input, "_text_to_hiragana_len", lambda text: 4)
    monkeypatch.setattr(ehr_input, "_capture_frame", lambda config: _make_frame())
    monkeypatch.setattr(ehr_input, "_has_ime_composition", lambda frame: next(composition_iter))

    ehr_input._cancel_ime_popup_safe(DummyClient(), "呼気性", config=config)

    # Esc + F6 + conservative floor BS×3 + Phase 2 VLM says remaining → BS×1
    assert events == [
        ("key", "escape"),
        ("key", "f6"),
        ("key", "backspace"),
        ("key", "backspace"),
        ("key", "backspace"),
        ("key", "backspace"),
    ]


def test_cancel_ime_popup_safe_vlm_false_negative_uses_conservative_bs(monkeypatch):
    """VLM が Esc+F6 直後に組成を検出できない場合、控えめな固定 BS で対応。"""
    events = []
    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)
    import numpy as np

    def _make_frame():
        return np.zeros((100, 200, 3), dtype=np.uint8)

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

    # VLM は常に False を返す（偽陰性）
    monkeypatch.setattr(ehr_input, "_text_to_hiragana_len", lambda text: 4)
    monkeypatch.setattr(ehr_input, "_capture_frame", lambda config: _make_frame())
    monkeypatch.setattr(ehr_input, "_has_ime_composition", lambda frame: False)

    ehr_input._cancel_ime_popup_safe(DummyClient(), "呼気性", config=config)

    # conservative = max(4-1, 1) = 3 fixed BS
    assert events == [
        ("key", "escape"),
        ("key", "f6"),
        ("key", "backspace"),
        ("key", "backspace"),
        ("key", "backspace"),
    ]


def test_find_best_candidate_match_strict_rejects_fuzzy():
    """strict=True ではファジーマッチを無効にし、完全一致のみを許可する。
    組成残存で「呼気を」が「呼応を」にファジーマッチするケースを防ぐ。"""
    candidates = [(1, "子機を"), (2, "古希を"), (6, "呼気を"), (9, "こきを")]
    # fuzzy (default): 呼気を matches 呼応を (same-length, 1 mismatch, 3 chars)
    result = ehr_input._find_best_candidate_match("呼応を", candidates)
    assert result is not None
    assert result[1] == "呼気を"
    # strict: no fuzzy match → None
    result_strict = ehr_input._find_best_candidate_match("呼応を", candidates, strict=True)
    assert result_strict is None


def test_find_best_candidate_match_strict_allows_exact():
    """strict=True でも完全一致は許可する。"""
    candidates = [(1, "子機"), (3, "呼応")]
    result = ehr_input._find_best_candidate_match("呼応", candidates, strict=True)
    assert result == (3, "呼応")


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


def test_cancel_ime_popup_safe_extra_budget_increases_bs(monkeypatch):
    """extra_budget が指定された場合、BS 回数が増加する（汚染キャンセル用）。"""
    events = []
    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)
    import numpy as np

    def _make_frame():
        return np.zeros((100, 200, 3), dtype=np.uint8)

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

    # VLM false negative path; hira_len=4 + extra_budget=4 = 8, conservative=max(8-1,1)=7
    monkeypatch.setattr(ehr_input, "_romaji_to_hiragana_len", lambda r: 4)
    monkeypatch.setattr(ehr_input, "_capture_frame", lambda config: _make_frame())
    monkeypatch.setattr(ehr_input, "_has_ime_composition", lambda frame: False)

    ehr_input._cancel_ime_popup_safe(
        DummyClient(), "著者", config=config, romaji="chosha", extra_budget=4,
    )

    # Esc + F6 + conservative BS×7
    bs_count = sum(1 for ev in events if ev == ("key", "backspace"))
    assert bs_count == 7


def test_is_helper_popup_contaminated_detects_residual():
    """組成残存で候補が汚染されている場合に True を返す。"""
    helper_word = "著者"
    numbered = [(1, "ちょめ"), (2, "緒"), (3, "著"), (4, "貯"), (5, "チョメ")]
    # Normalization returns the original (nothing matches "著者")
    normalized = numbered
    assert ehr_input._is_helper_popup_contaminated(helper_word, numbered, normalized) is True


def test_is_helper_popup_contaminated_clean_popup():
    """正常な候補リストでは False を返す。"""
    helper_word = "著者"
    numbered = [(1, "著者"), (2, "ちょしゃ"), (3, "チョシャ")]
    # Normalization filters to just "著者"
    normalized = [(1, "著者")]
    assert ehr_input._is_helper_popup_contaminated(helper_word, numbered, normalized) is False


def test_is_helper_popup_contaminated_empty_candidates():
    """空の候補リストでは False を返す。"""
    assert ehr_input._is_helper_popup_contaminated("著者", [], []) is False


def test_is_helper_popup_contaminated_first_char_matches():
    """第1候補の先頭文字がヘルパーと一致する場合は汚染とみなさない。"""
    helper_word = "著者"
    numbered = [(1, "著作"), (2, "著名")]
    normalized = numbered  # Nothing filtered
    assert ehr_input._is_helper_popup_contaminated(helper_word, numbered, normalized) is False


def test_is_helper_popup_contaminated_kanji_mismatch_is_ocr_noise():
    """第1候補が漢字でヘルパーと不一致でも、OCRノイズであり汚染とみなさない。"""
    helper_word = "過剰"
    # OCR misreads "過剰" as "箇条" — kanji, not hiragana
    numbered = [(2, "箇条"), (3, "潟状"), (5, "家京")]
    normalized = numbered
    assert ehr_input._is_helper_popup_contaminated(helper_word, numbered, normalized) is False


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


def test_reset_ime_before_helper_lookup_stops_when_vlm_says_ready(monkeypatch):
    events = []
    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)
    frame = _make_frame()

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

    states = iter([
        {"left_context_preserved": True, "composition_cleared": False, "ready": False},
        {"left_context_preserved": True, "composition_cleared": True, "ready": True},
    ])
    monkeypatch.setattr(ehr_input, "_capture_frame", lambda config: frame)
    captured = []
    monkeypatch.setattr(
        ehr_input,
        "assess_helper_reset_state",
        lambda frame, left_context, anchor_text, target_text: (
            captured.append((left_context, anchor_text, target_text)) or next(states)
        ),
    )

    assert ehr_input._reset_ime_before_helper_lookup(
        DummyClient(),
        config,
        target_kanji="咽頭痛",
        left_context="昨日から",
        anchor_text="咽頭",
    )
    assert events == [("key", "escape"), ("key", "escape")]
    assert captured == [("昨日から", "咽頭", "咽頭痛"), ("昨日から", "咽頭", "咽頭痛")]


def test_reset_ime_before_helper_lookup_stops_when_compare_matches_baseline(monkeypatch):
    events = []
    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)
    frame = _make_frame()
    baseline_crop = np.zeros((40, 120, 3), dtype=np.uint8)
    current_crop = np.ones((40, 120, 3), dtype=np.uint8)

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

    states = iter([False, True])
    monkeypatch.setattr(ehr_input, "_capture_frame", lambda config: frame)
    monkeypatch.setattr(
        ehr_input,
        "_capture_helper_reset_compare_frame",
        lambda frame, debug_name="", screen_type=None: current_crop,
    )
    captured = []
    monkeypatch.setattr(
        ehr_input,
        "compare_helper_reset_images",
        lambda baseline, current, left_context="", anchor_text="", target_text="", screen_type="": (
            captured.append((baseline, current, left_context, anchor_text, target_text, screen_type)) or next(states)
        ),
    )

    assert ehr_input._reset_ime_before_helper_lookup(
        DummyClient(),
        config,
        target_kanji="痛",
        left_context="昨日から感冒症状(",
        anchor_text="咽頭(",
        baseline_state={"cropped_frame": baseline_crop, "anchor_text": "咽頭(", "screen_type": "notepad"},
    )
    assert events == [("key", "escape"), ("key", "escape")]
    trimmed_left_context = ehr_input._trim_helper_left_context("昨日から感冒症状(")
    assert captured == [
        (baseline_crop, current_crop, trimmed_left_context, "咽頭(", "痛", "notepad"),
        (baseline_crop, current_crop, trimmed_left_context, "咽頭(", "痛", "notepad"),
    ]


def test_reset_ime_before_helper_lookup_fails_after_compare_never_matches(monkeypatch):
    events = []
    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)
    frame = _make_frame()
    baseline_crop = np.zeros((40, 120, 3), dtype=np.uint8)
    current_crop = np.ones((40, 120, 3), dtype=np.uint8)

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

    monkeypatch.setattr(ehr_input, "_capture_frame", lambda config: frame)
    monkeypatch.setattr(
        ehr_input,
        "_capture_helper_reset_compare_frame",
        lambda frame, debug_name="", screen_type=None: current_crop,
    )
    monkeypatch.setattr(
        ehr_input,
        "compare_helper_reset_images",
        lambda baseline, current, left_context="", anchor_text="", target_text="", screen_type="": False,
    )

    assert not ehr_input._reset_ime_before_helper_lookup(
        DummyClient(),
        config,
        target_kanji="痛",
        anchor_text="咽頭",
        baseline_state={"cropped_frame": baseline_crop, "anchor_text": "咽頭", "screen_type": "notepad"},
    )
    assert events == [("key", "escape"), ("key", "escape"), ("key", "escape"), ("key", "escape")]


def test_reset_ime_before_helper_lookup_logs_when_compare_requests_next_escape(monkeypatch, capsys):
    events = []
    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)
    frame = _make_frame()
    baseline_crop = np.zeros((40, 120, 3), dtype=np.uint8)
    current_crop = np.ones((40, 120, 3), dtype=np.uint8)

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

    monkeypatch.setattr(ehr_input, "_capture_frame", lambda config: frame)
    monkeypatch.setattr(
        ehr_input,
        "_capture_helper_reset_compare_frame",
        lambda frame, debug_name="", screen_type=None: current_crop,
    )
    monkeypatch.setattr(
        ehr_input,
        "compare_helper_reset_images",
        lambda baseline, current, left_context="", anchor_text="", target_text="", screen_type="": False,
    )

    assert not ehr_input._reset_ime_before_helper_lookup(
        DummyClient(),
        config,
        target_kanji="痛",
        anchor_text="咽頭",
        baseline_state={"cropped_frame": baseline_crop, "anchor_text": "咽頭", "screen_type": "notepad"},
        max_escape_count=2,
    )

    output = capsys.readouterr().out
    assert "baseline_vs_current=no" in output
    assert "リセット未完了 → 次の Escape を送ります" in output


def test_reset_ime_before_helper_lookup_aborts_when_left_context_breaks(monkeypatch):
    events = []
    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)
    frame = _make_frame()

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

    monkeypatch.setattr(ehr_input, "_capture_frame", lambda config: frame)
    monkeypatch.setattr(
        ehr_input,
        "assess_helper_reset_state",
        lambda frame, left_context, anchor_text, target_text: {
            "left_context_preserved": False,
            "composition_cleared": False,
            "ready": False,
        },
    )

    assert not ehr_input._reset_ime_before_helper_lookup(
        DummyClient(),
        config,
        target_kanji="咽頭痛",
        left_context="昨日から",
        anchor_text="咽頭",
    )
    assert events == [("key", "escape")]


def test_reset_ime_before_helper_lookup_logs_when_state_requests_next_escape(monkeypatch, capsys):
    events = []
    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)
    frame = _make_frame()

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

    states = iter(
        [
            {
                "left_context_preserved": True,
                "composition_cleared": False,
                "ready": False,
            },
            {
                "left_context_preserved": True,
                "composition_cleared": True,
                "ready": True,
            },
        ]
    )
    monkeypatch.setattr(ehr_input, "_capture_frame", lambda config: frame)
    monkeypatch.setattr(ehr_input, "assess_helper_reset_state", lambda *args, **kwargs: next(states))

    assert ehr_input._reset_ime_before_helper_lookup(
        DummyClient(),
        config,
        target_kanji="咽頭痛",
        left_context="昨日から",
        anchor_text="咽頭",
        max_escape_count=2,
    )

    output = capsys.readouterr().out
    assert "ready=False" in output
    assert "リセット未完了 → 次の Escape を送ります" in output
    assert "ready=true → Escape 停止" in output


def test_assess_helper_reset_state_prompt_uses_anchor_text(monkeypatch):
    frame = _make_frame()
    captured = {}

    monkeypatch.setattr(mlx_vlm_ime, "crop_helper_reset_region", lambda image, **kwargs: (image, "patient_record"))
    monkeypatch.setattr(mlx_vlm_ime, "_crop_center_band", lambda image, **kwargs: image)
    monkeypatch.setattr(mlx_vlm_ime, "_encode_image_data_url", lambda image, **kwargs: "data:image/mock")

    def _fake_call(data_url, prompt, timeout):
        captured["prompt"] = prompt
        return '{"left_context_preserved": true, "composition_cleared": false, "ready": false}'

    monkeypatch.setattr(mlx_vlm_ime, "_call_mlx_vlm_with_image", _fake_call)

    state = mlx_vlm_ime.assess_helper_reset_state(
        frame,
        left_context="昨日から感冒症状(",
        anchor_text="咽頭",
        target_text="痛",
    )

    assert state == {
        "left_context_preserved": True,
        "composition_cleared": False,
        "ready": False,
    }
    assert "確定済みアンカー文字列として '咽頭'" in captured["prompt"]
    assert "'咽頭' の直後に、'痛'" in captured["prompt"]


def test_detect_patient_record_panel3_finds_third_pane_from_four_gray_dividers():
    frame = _make_patient_record_frame()

    assert mlx_vlm_ime.detect_patient_record_panel3(frame) == (200, 300)


def test_detect_patient_record_panel3_saves_opencv_overlay(monkeypatch):
    frame = _make_patient_record_frame()
    saved = []

    monkeypatch.setattr(
        mlx_vlm_ime,
        "_save_debug_frame",
        lambda image, *, name, prefix: saved.append((image.shape, name, prefix)),
    )

    assert mlx_vlm_ime.detect_patient_record_panel3(frame, debug_name="helper_reset_panel") == (200, 300)
    assert saved == [((240, 600, 3), "helper_reset_panel", "debug_panel_detection")]


def test_assess_helper_reset_state_requests_panel_and_vlm_debug(monkeypatch):
    frame = _make_frame()
    calls = []

    monkeypatch.setattr(
        mlx_vlm_ime,
        "crop_helper_reset_region",
        lambda image, **kwargs: calls.append(("crop", kwargs.get("debug_name"))) or (image, "patient_record"),
    )
    monkeypatch.setattr(mlx_vlm_ime, "_crop_center_band", lambda image, **kwargs: image)
    monkeypatch.setattr(
        mlx_vlm_ime,
        "_encode_image_data_url",
        lambda image, **kwargs: calls.append(("encode", kwargs.get("debug_name"))) or "data:image/mock",
    )
    monkeypatch.setattr(
        mlx_vlm_ime,
        "_call_mlx_vlm_with_image",
        lambda data_url, prompt, timeout: '{"left_context_preserved": true, "composition_cleared": true, "ready": true}',
    )

    state = mlx_vlm_ime.assess_helper_reset_state(
        frame,
        left_context="昨日から",
        anchor_text="咽頭",
        target_text="痛",
    )

    assert state["ready"] is True
    assert calls == [("crop", "helper_reset_panel"), ("encode", "helper_reset_state")]


def test_capture_helper_reset_baseline_stores_screen_type(monkeypatch):
    frame = _make_frame()
    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)

    monkeypatch.setattr(ehr_input, "_capture_frame", lambda config: frame)
    monkeypatch.setattr(
        mlx_vlm_ime,
        "classify_helper_reset_screen",
        lambda image, debug_name="": "notepad",
    )
    monkeypatch.setattr(
        ehr_input,
        "_capture_helper_reset_compare_frame",
        lambda image, debug_name="", screen_type=None: np.ones((20, 40, 3), dtype=np.uint8),
    )

    state = ehr_input._capture_helper_reset_baseline(config, anchor_text="咽頭", debug_name="helper_reset")

    assert state is not None
    assert state["anchor_text"] == "咽頭"
    assert state["screen_type"] == "notepad"
    assert state["cropped_frame"].shape == (20, 40, 3)


def test_prime_helper_reset_panel_cache_captures_once(monkeypatch):
    frame = _make_frame()
    config = SimpleNamespace(capture_device_index=0, capture_width=1920, capture_height=1080)
    calls = []

    monkeypatch.setattr(mlx_vlm_ime, "reset_helper_reset_panel_cache", lambda: calls.append(("reset",)))
    monkeypatch.setattr(ehr_input, "_capture_frame", lambda config: frame)
    monkeypatch.setattr(
        mlx_vlm_ime,
        "prime_helper_reset_panel_cache",
        lambda image, debug_name="": calls.append(("prime", image.shape, debug_name)),
    )

    ehr_input._prime_helper_reset_panel_cache(config)

    assert calls == [
        ("reset",),
        ("prime", frame.shape, "helper_reset_initial_panel"),
    ]


def test_capture_helper_reset_compare_frame_focuses_patient_record_center_band(monkeypatch):
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    panel_crop = np.ones((100, 80, 3), dtype=np.uint8)
    center_band = np.full((50, 80, 3), 7, dtype=np.uint8)

    monkeypatch.setattr(
        mlx_vlm_ime,
        "crop_helper_reset_region",
        lambda image, **kwargs: (panel_crop, "patient_record"),
    )
    monkeypatch.setattr(
        mlx_vlm_ime,
        "_crop_center_band",
        lambda image, **kwargs: center_band,
    )

    cropped = ehr_input._capture_helper_reset_compare_frame(
        frame,
        debug_name="helper_reset",
        screen_type="patient_record",
    )

    assert cropped is center_band


def test_compare_helper_reset_images_prompt_uses_anchor_and_two_images(monkeypatch):
    baseline = _make_frame()
    current = _make_frame()
    calls = []
    captured = {}

    monkeypatch.setattr(
        mlx_vlm_ime,
        "_encode_image_data_url",
        lambda image, **kwargs: calls.append(("encode", kwargs.get("debug_name"))) or f"data:image/{len(calls)}",
    )

    def _fake_call(data_urls, prompt, timeout, enable_reasoning=False, thinking_log=False):
        captured["data_urls"] = data_urls
        captured["prompt"] = prompt
        captured["enable_reasoning"] = enable_reasoning
        captured["thinking_log"] = thinking_log
        return "yes"

    monkeypatch.setattr(mlx_vlm_ime, "_call_mlx_vlm_with_images", _fake_call)

    assert mlx_vlm_ime.compare_helper_reset_images(
        baseline,
        current,
        anchor_text="咽頭",
        target_text="痛",
        left_context="昨日から感冒症状(",
        screen_type="notepad",
    ) is True
    assert calls == [
        ("encode", "helper_reset_compare_baseline"),
        ("encode", "helper_reset_compare_current"),
    ]
    assert captured["data_urls"] == ["data:image/1", "data:image/2"]
    assert captured["enable_reasoning"] is True
    assert captured["thinking_log"] is True
    assert "ここでは2枚の別画像が送られます" in captured["prompt"]
    assert "1枚目を縦に分割した1枚画像だと思わないでください" in captured["prompt"]
    assert "1枚目の最後の確定済み文字列は '咽頭'" in captured["prompt"]
    assert "Windows Notepad の本文領域です" in captured["prompt"]
    assert "yes または no のみで答えてください" in captured["prompt"]


def test_compare_helper_reset_images_prompt_accepts_anchor_with_ascii_suffix(monkeypatch):
    baseline = _make_frame()
    current = _make_frame()
    captured = {}

    monkeypatch.setattr(
        mlx_vlm_ime,
        "_encode_image_data_url",
        lambda image, **kwargs: "data:image/mock",
    )
    def _fake_call(data_urls, prompt, timeout, enable_reasoning=False, thinking_log=False):
        captured["prompt"] = prompt
        return "yes"

    monkeypatch.setattr(mlx_vlm_ime, "_call_mlx_vlm_with_images", _fake_call)

    assert mlx_vlm_ime.compare_helper_reset_images(
        baseline,
        current,
        anchor_text="症状(",
        target_text="咽頭痛",
        left_context="感冒症状(",
        screen_type="notepad",
    ) is True
    assert "1枚目の最後の確定済み文字列は '症状('" in captured["prompt"]
    assert "'症状(' の後ろに別の文字" in captured["prompt"]


def test_type_japanese_sentence_routes_japanese_punctuation_via_ascii_keys(monkeypatch):
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

    ehr_input.type_japanese_sentence("、")

    assert events == [("type", ","), ("key", "enter")]


def test_type_japanese_sentence_always_confirms_japanese_comma_before_next_segment(monkeypatch):
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
    monkeypatch.setattr(ehr_input.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        ehr_input,
        "_iter_segments_for_input",
        lambda text: iter([
            {"text": "、", "romaji": ","},
            {"text": "強制", "romaji": "kyousei"},
        ]),
    )
    monkeypatch.setattr(
        ehr_input,
        "type_kanji_via_ime",
        lambda romaji, target, **kwargs: events.append(("ime", romaji, target, kwargs)),
    )
    monkeypatch.setattr(ehr_input, "_capture_helper_reset_baseline", lambda *args, **kwargs: None)

    ehr_input.type_japanese_sentence("、強制")

    assert events == [
        ("type", ","),
        ("key", "enter"),
        ("ime", "kyousei", "強制", {"_current_ime_mode": "japanese", "_typed_prefix_context": "、", "_helper_anchor_text": "", "_helper_reset_baseline": None}),
    ]


def test_type_japanese_sentence_always_confirms_japanese_period_before_next_segment(monkeypatch):
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
    monkeypatch.setattr(ehr_input.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        ehr_input,
        "_iter_segments_for_input",
        lambda text: iter([
            {"text": "。", "romaji": "."},
            {"text": "強制", "romaji": "kyousei"},
        ]),
    )
    monkeypatch.setattr(
        ehr_input,
        "type_kanji_via_ime",
        lambda romaji, target, **kwargs: events.append(("ime", romaji, target, kwargs)),
    )
    monkeypatch.setattr(ehr_input, "_capture_helper_reset_baseline", lambda *args, **kwargs: None)

    ehr_input.type_japanese_sentence("。強制")

    assert events == [
        ("type", "."),
        ("key", "enter"),
        ("ime", "kyousei", "強制", {"_current_ime_mode": "japanese", "_typed_prefix_context": "。", "_helper_anchor_text": "", "_helper_reset_baseline": None}),
    ]


def test_type_japanese_sentence_always_confirms_japanese_middle_dot_before_next_segment(monkeypatch):
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
    monkeypatch.setattr(ehr_input.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        ehr_input,
        "_iter_segments_for_input",
        lambda text: iter([
            {"text": "・", "romaji": "/"},
            {"text": "コーテフ", "romaji": "ko-tefu"},
        ]),
    )
    monkeypatch.setattr(
        ehr_input,
        "_katakana_to_romaji",
        lambda text: {"コーテフ": "ko-tefu"}[text],
    )
    monkeypatch.setattr(ehr_input, "_capture_helper_reset_baseline", lambda *args, **kwargs: None)

    ehr_input.type_japanese_sentence("・コーテフ")

    assert events == [
        ("type", "/"),
        ("key", "enter"),
        ("type", "ko-tefu"),
        ("key", "f7"),
        ("key", "enter"),
    ]


def test_type_japanese_sentence_always_confirms_long_vowel_mark_before_next_segment(monkeypatch):
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
    monkeypatch.setattr(ehr_input.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        ehr_input,
        "_iter_segments_for_input",
        lambda text: iter([
            {"text": "ー", "romaji": "-"},
            {"text": "強制", "romaji": "kyousei"},
        ]),
    )
    monkeypatch.setattr(
        ehr_input,
        "type_kanji_via_ime",
        lambda romaji, target, **kwargs: events.append(("ime", romaji, target, kwargs)),
    )
    monkeypatch.setattr(ehr_input, "_capture_helper_reset_baseline", lambda *args, **kwargs: None)

    ehr_input.type_japanese_sentence("ー強制")

    assert events == [
        ("type", "-"),
        ("key", "enter"),
        ("ime", "kyousei", "強制", {"_current_ime_mode": "japanese", "_typed_prefix_context": "ー", "_helper_anchor_text": "", "_helper_reset_baseline": None}),
    ]


def test_type_japanese_sentence_always_confirms_wave_dash_before_next_segment(monkeypatch):
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
    monkeypatch.setattr(ehr_input.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        ehr_input,
        "_iter_segments_for_input",
        lambda text: iter([
            {"text": "〜", "romaji": "~"},
            {"text": "強制", "romaji": "kyousei"},
        ]),
    )
    monkeypatch.setattr(
        ehr_input,
        "type_kanji_via_ime",
        lambda romaji, target, **kwargs: events.append(("ime", romaji, target, kwargs)),
    )
    monkeypatch.setattr(ehr_input, "_capture_helper_reset_baseline", lambda *args, **kwargs: None)

    ehr_input.type_japanese_sentence("〜強制")

    assert events == [
        ("type", "~"),
        ("key", "enter"),
        ("ime", "kyousei", "強制", {"_current_ime_mode": "japanese", "_typed_prefix_context": "〜", "_helper_anchor_text": "", "_helper_reset_baseline": None}),
    ]


def test_type_japanese_sentence_confirms_japanese_bracket_before_next_conversion(monkeypatch):
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
    monkeypatch.setattr(ehr_input.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        ehr_input,
        "_iter_segments_for_input",
        lambda text: iter([
            {"text": "「", "romaji": "["},
            {"text": "過", "romaji": "ka"},
        ]),
    )
    monkeypatch.setattr(
        ehr_input,
        "type_kanji_via_ime",
        lambda romaji, target, **kwargs: events.append(("ime", romaji, target, kwargs)),
    )
    monkeypatch.setattr(ehr_input, "_capture_helper_reset_baseline", lambda *args, **kwargs: None)

    ehr_input.type_japanese_sentence("「過")

    assert events == [
        ("key", "lbracket"),
        ("key", "enter"),
        ("ime", "ka", "過", {"_current_ime_mode": "japanese", "_typed_prefix_context": "「", "_helper_anchor_text": "", "_helper_reset_baseline": None}),
    ]


def test_type_japanese_sentence_passes_prefix_context_to_kanji_segments(monkeypatch):
    events = []

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

        def type_text(self, text):
            events.append(("type", text))
            return True

    monkeypatch.setattr(ehr_input, "load_config", lambda skip_password=True: SimpleNamespace())
    monkeypatch.setattr(ehr_input, "_wait_for_ble_connected", lambda: DummyClient())
    monkeypatch.setattr(ehr_input, "detect_ime_mode", lambda *args, **kwargs: "japanese")
    monkeypatch.setattr(ehr_input, "ensure_ime_mode", lambda target, client, current: target)
    monkeypatch.setattr(
        ehr_input,
        "_iter_segments_for_input",
        lambda text: iter([
            {"text": "から", "romaji": "kara"},
            {"text": "咽頭痛", "romaji": "intoutsuu"},
        ]),
    )
    monkeypatch.setattr(
        ehr_input,
        "type_kanji_via_ime",
        lambda romaji, target, **kwargs: events.append(("ime", romaji, target, kwargs)),
    )
    monkeypatch.setattr(ehr_input, "_capture_helper_reset_baseline", lambda *args, **kwargs: None)

    ehr_input.type_japanese_sentence("から咽頭痛")

    assert (
        "ime",
        "intoutsuu",
        "咽頭痛",
        {
            "_current_ime_mode": "japanese",
            "_typed_prefix_context": "から",
            "_helper_anchor_text": "から",
            "_helper_reset_baseline": None,
        },
    ) in events


def test_type_japanese_sentence_uses_previous_confirmed_segment_as_helper_anchor(monkeypatch):
    events = []

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

        def type_text(self, text):
            events.append(("type", text))
            return True

    monkeypatch.setattr(ehr_input, "load_config", lambda skip_password=True: SimpleNamespace())
    monkeypatch.setattr(ehr_input, "_wait_for_ble_connected", lambda: DummyClient())
    monkeypatch.setattr(ehr_input, "detect_ime_mode", lambda *args, **kwargs: "japanese")
    monkeypatch.setattr(ehr_input, "ensure_ime_mode", lambda target, client, current: target)
    monkeypatch.setattr(
        ehr_input,
        "_iter_segments_for_input",
        lambda text: iter([
            {"text": "咽頭", "romaji": "intou"},
            {"text": "痛", "romaji": "ita"},
        ]),
    )
    monkeypatch.setattr(
        ehr_input,
        "_capture_helper_reset_baseline",
        lambda config, anchor_text="", debug_name="helper_reset_baseline": (
            {"final_line": "咽頭", "char_count": 2, "anchor_present": True}
            if anchor_text == "咽頭"
            else None
        ),
    )
    monkeypatch.setattr(
        ehr_input,
        "type_kanji_via_ime",
        lambda romaji, target, **kwargs: events.append(("ime", romaji, target, kwargs)),
    )

    ehr_input.type_japanese_sentence("咽頭痛")

    assert events == [
        ("ime", "intou", "咽頭", {"_current_ime_mode": "japanese", "_typed_prefix_context": "", "_helper_anchor_text": "", "_helper_reset_baseline": None}),
        ("ime", "ita", "痛", {"_current_ime_mode": "japanese", "_typed_prefix_context": "咽頭", "_helper_anchor_text": "咽頭", "_helper_reset_baseline": {"final_line": "咽頭", "char_count": 2, "anchor_present": True}}),
    ]


def test_type_japanese_sentence_carries_ascii_suffix_into_helper_anchor(monkeypatch):
    events = []
    baseline_calls = []

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

        def type_text(self, text):
            events.append(("type", text))
            return True

    monkeypatch.setattr(ehr_input, "load_config", lambda skip_password=True: SimpleNamespace())
    monkeypatch.setattr(ehr_input, "_wait_for_ble_connected", lambda: DummyClient())
    monkeypatch.setattr(ehr_input, "detect_ime_mode", lambda *args, **kwargs: "japanese")
    monkeypatch.setattr(ehr_input, "ensure_ime_mode", lambda target, client, current: target)
    monkeypatch.setattr(
        ehr_input,
        "_iter_segments_for_input",
        lambda text: iter([
            {"text": "症状", "romaji": "shoujou"},
            {"text": "(", "romaji": "("},
            {"text": "咽頭痛", "romaji": "intoutsuu"},
        ]),
    )
    monkeypatch.setattr(
        ehr_input,
        "_capture_helper_reset_baseline",
        lambda config, anchor_text="", debug_name="helper_reset_baseline": (
            baseline_calls.append(anchor_text) or {"anchor_text": anchor_text}
        ) if anchor_text else None,
    )
    monkeypatch.setattr(
        ehr_input,
        "type_kanji_via_ime",
        lambda romaji, target, **kwargs: events.append(("ime", romaji, target, kwargs)),
    )

    ehr_input.type_japanese_sentence("症状(咽頭痛")

    assert events == [
        ("ime", "shoujou", "症状", {"_current_ime_mode": "japanese", "_typed_prefix_context": "", "_helper_anchor_text": "", "_helper_reset_baseline": None}),
        ("key", "lparen"),
        ("ime", "intoutsuu", "咽頭痛", {"_current_ime_mode": "japanese", "_typed_prefix_context": "症状(", "_helper_anchor_text": "症状(", "_helper_reset_baseline": {"anchor_text": "症状("}}),
    ]
    assert baseline_calls == ["症状("]


def test_type_japanese_sentence_routes_katakana_segment_via_f7_before_kanji(monkeypatch):
    events = []
    baseline_calls = []

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

        def type_text(self, text):
            events.append(("type", text))
            return True

    monkeypatch.setattr(ehr_input, "load_config", lambda skip_password=True: SimpleNamespace())
    monkeypatch.setattr(ehr_input, "_wait_for_ble_connected", lambda: DummyClient())
    monkeypatch.setattr(ehr_input, "detect_ime_mode", lambda *args, **kwargs: "japanese")
    monkeypatch.setattr(ehr_input, "ensure_ime_mode", lambda target, client, current: target)
    monkeypatch.setattr(ehr_input.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        ehr_input,
        "_iter_segments_for_input",
        lambda text: iter([
            {"text": "アレルギー", "romaji": "arerugi-"},
            {"text": "性", "romaji": "sei"},
        ]),
    )
    monkeypatch.setattr(
        ehr_input,
        "type_kanji_via_ime",
        lambda romaji, target, **kwargs: events.append(("ime", romaji, target, kwargs)),
    )
    monkeypatch.setattr(
        ehr_input,
        "_capture_helper_reset_baseline",
        lambda config, anchor_text="", debug_name="helper_reset_baseline": (
            baseline_calls.append(anchor_text) or {"anchor_text": anchor_text}
        ) if anchor_text else None,
    )

    ehr_input.type_japanese_sentence("アレルギー性")

    assert events == [
        ("type", "arerugi-"),
        ("key", "f7"),
        ("key", "enter"),
        ("ime", "sei", "性", {"_current_ime_mode": "japanese", "_typed_prefix_context": "アレルギー", "_helper_anchor_text": "アレルギー", "_helper_reset_baseline": {"anchor_text": "アレルギー"}}),
    ]
    assert baseline_calls == ["アレルギー"]


def test_type_japanese_sentence_skips_baseline_when_upcoming_segments_do_not_use_ime(monkeypatch):
    events = []
    baseline_calls = []

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

        def type_text(self, text):
            events.append(("type", text))
            return True

    monkeypatch.setattr(ehr_input, "load_config", lambda skip_password=True: SimpleNamespace())
    monkeypatch.setattr(ehr_input, "_wait_for_ble_connected", lambda: DummyClient())
    monkeypatch.setattr(ehr_input, "detect_ime_mode", lambda *args, **kwargs: "japanese")
    monkeypatch.setattr(ehr_input, "ensure_ime_mode", lambda target, client, current: target)
    monkeypatch.setattr(ehr_input.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        ehr_input,
        "_iter_segments_for_input",
        lambda text: iter([
            {"text": "症状", "romaji": "shoujou"},
            {"text": "(", "romaji": "("},
            {"text": "CRP", "romaji": "CRP"},
        ]),
    )
    monkeypatch.setattr(
        ehr_input,
        "_capture_helper_reset_baseline",
        lambda config, anchor_text="", debug_name="helper_reset_baseline": (
            baseline_calls.append(anchor_text) or {"anchor_text": anchor_text}
        ) if anchor_text else None,
    )
    monkeypatch.setattr(
        ehr_input,
        "type_kanji_via_ime",
        lambda romaji, target, **kwargs: events.append(("ime", romaji, target, kwargs)),
    )

    ehr_input.type_japanese_sentence("症状(CRP")

    assert events == [
        ("ime", "shoujou", "症状", {"_current_ime_mode": "japanese", "_typed_prefix_context": "", "_helper_anchor_text": "", "_helper_reset_baseline": None}),
        ("key", "lparen"),
        ("type", "CRP"),
    ]
    assert baseline_calls == []


def test_segment_japanese_with_openrouter_uses_runtime_aware_logs(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "token-xyz")
    monkeypatch.setattr(
        ehr_input,
        "segment_japanese_text_with_mlx_vlm",
        lambda text: ('[{"text":"肺炎","romaji":"haien"}]', [{"text": "肺炎", "romaji": "haien"}]),
    )
    monkeypatch.setattr(ehr_input, "_kanji_to_romaji", lambda text: "haien")
    stdout = io.StringIO()

    ehr_input._configure_runtime(openrouter_model="google/gemma-4-26b-a4b-it")
    try:
        with redirect_stdout(stdout):
            assert ehr_input._segment_japanese_with_default_vlm("肺炎") == [
                {"text": "肺炎", "romaji": "haien"}
            ]
    finally:
        ehr_input._configure_runtime(openrouter_model=None)

    output = stdout.getvalue()
    assert "OpenRouter(google/gemma-4-26b-a4b-it)分割結果" in output
    assert "Qwen分割結果" not in output


def test_segment_japanese_with_google_ai_studio_uses_runtime_aware_logs(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "token-gemini")
    monkeypatch.setattr(
        ehr_input,
        "segment_japanese_text_with_mlx_vlm",
        lambda text: ('[{"text":"肺炎","romaji":"haien"}]', [{"text": "肺炎", "romaji": "haien"}]),
    )
    monkeypatch.setattr(ehr_input, "_kanji_to_romaji", lambda text: "haien")
    stdout = io.StringIO()

    ehr_input._configure_runtime(google_ai_studio=True)
    try:
        with redirect_stdout(stdout):
            assert ehr_input._segment_japanese_with_default_vlm("肺炎") == [
                {"text": "肺炎", "romaji": "haien"}
            ]
    finally:
        ehr_input._configure_runtime(openrouter_model=None, google_ai_studio=False)

    output = stdout.getvalue()
    assert "Google AI Studio(gemma-4-26b-a4b-it)分割結果" in output
    assert "Qwen分割結果" not in output


def test_segment_japanese_with_novita_uses_runtime_aware_logs(monkeypatch):
    monkeypatch.setenv("NOVITA_API_KEY", "token-novita")
    monkeypatch.setattr(
        ehr_input,
        "segment_japanese_text_with_mlx_vlm",
        lambda text: ('[{"text":"肺炎","romaji":"haien"}]', [{"text": "肺炎", "romaji": "haien"}]),
    )
    monkeypatch.setattr(ehr_input, "_kanji_to_romaji", lambda text: "haien")
    stdout = io.StringIO()

    ehr_input._configure_runtime(novita_model="google/gemma-4-31b-it")
    try:
        with redirect_stdout(stdout):
            assert ehr_input._segment_japanese_with_default_vlm("肺炎") == [
                {"text": "肺炎", "romaji": "haien"}
            ]
    finally:
        ehr_input._configure_runtime(openrouter_model=None, novita_model=None, google_ai_studio=False)

    output = stdout.getvalue()
    assert "Novita(google/gemma-4-31b-it)分割結果" in output
    assert "Qwen分割結果" not in output


def test_dual_provider_wrappers_alternate_between_openrouter_and_novita(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "token-openrouter")
    monkeypatch.setenv("NOVITA_API_KEY", "token-novita")
    calls = []

    monkeypatch.setattr(
        ehr_input,
        "_segment_japanese_text_with_mlx_vlm_impl",
        lambda text: calls.append(
            (
                "segmentation",
                ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_URL,
                ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_MODEL,
                ehr_input.mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_API_KEY,
            )
        ) or ("[]", []),
    )
    monkeypatch.setattr(
        ehr_input,
        "_suggest_ime_helper_word_impl",
        lambda text: calls.append(
            (
                "text",
                ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_URL,
                ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_MODEL,
                ehr_input.mlx_vlm_ime.MLX_VLM_TEXT_API_KEY,
            )
        ) or [],
    )

    ehr_input._configure_runtime(
        openrouter_model="google/gemma-4-31b-it",
        novita_model="google/gemma-4-31b-it",
    )
    try:
        ehr_input.segment_japanese_text_with_mlx_vlm("肺炎")
        ehr_input.suggest_ime_helper_word("肺")
        ehr_input.segment_japanese_text_with_mlx_vlm("炎")
    finally:
        ehr_input._configure_runtime(openrouter_model=None, novita_model=None, google_ai_studio=False)

    assert calls == [
        (
            "segmentation",
            "https://openrouter.ai/api/v1/chat/completions",
            "google/gemma-4-31b-it",
            "token-openrouter",
        ),
        (
            "text",
            "https://api.novita.ai/openai/chat/completions",
            "google/gemma-4-31b-it",
            "token-novita",
        ),
        (
            "segmentation",
            "https://openrouter.ai/api/v1/chat/completions",
            "google/gemma-4-31b-it",
            "token-openrouter",
        ),
    ]


def test_segment_japanese_with_fireworks_uses_runtime_aware_logs(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "token-fireworks")
    monkeypatch.setattr(
        ehr_input,
        "segment_japanese_text_with_mlx_vlm",
        lambda text: ('[{"text":"肺炎","romaji":"haien"}]', [{"text": "肺炎", "romaji": "haien"}]),
    )
    monkeypatch.setattr(ehr_input, "_kanji_to_romaji", lambda text: "haien")
    stdout = io.StringIO()

    ehr_input._configure_runtime(fireworks_model="accounts/fireworks/models/gemma-4-26b-a4b-it")
    try:
        with redirect_stdout(stdout):
            assert ehr_input._segment_japanese_with_default_vlm("肺炎") == [
                {"text": "肺炎", "romaji": "haien"}
            ]
    finally:
        ehr_input._configure_runtime(openrouter_model=None, fireworks_model=None, google_ai_studio=False)

    output = stdout.getvalue()
    assert "Fireworks(accounts/fireworks/models/gemma-4-26b-a4b-it)分割結果" in output
    assert "Qwen分割結果" not in output


def test_parse_cli_options_rejects_google_ai_studio_with_openrouter():
    with pytest.raises(RuntimeError, match="同時に指定できません"):
        ehr_input._parse_cli_options(["--google-ai-studio", "--openrouter", "qwen/qwen3.5-9b", "肺炎"])


def test_parse_cli_options_parses_novita_default_model():
    args, option_summary = ehr_input._parse_cli_options(["--novita", "肺炎"])

    assert args == ["肺炎"]
    assert option_summary == {
        "clear_field": False,
        "dual_provider_mode": False,
        "fireworks_model": None,
        "google_ai_studio": False,
        "novita_model": "google/gemma-4-31b-it",
        "openrouter_model": None,
    }


def test_parse_cli_options_parses_novita_custom_model():
    args, option_summary = ehr_input._parse_cli_options(["--novita", "deepseek/deepseek-vl2", "肺炎"])

    assert args == ["肺炎"]
    assert option_summary == {
        "clear_field": False,
        "dual_provider_mode": False,
        "fireworks_model": None,
        "google_ai_studio": False,
        "novita_model": "deepseek/deepseek-vl2",
        "openrouter_model": None,
    }


def test_parse_cli_options_parses_novita_inline_model():
    args, option_summary = ehr_input._parse_cli_options(["--novita=google/gemma-4-31b-it", "肺炎"])

    assert args == ["肺炎"]
    assert option_summary == {
        "clear_field": False,
        "dual_provider_mode": False,
        "fireworks_model": None,
        "google_ai_studio": False,
        "novita_model": "google/gemma-4-31b-it",
        "openrouter_model": None,
    }


def test_parse_cli_options_parses_dual_provider_default_model():
    args, option_summary = ehr_input._parse_cli_options(["--openrouter", "--novita", "肺炎"])

    assert args == ["肺炎"]
    assert option_summary == {
        "clear_field": False,
        "dual_provider_mode": True,
        "fireworks_model": None,
        "google_ai_studio": False,
        "novita_model": "google/gemma-4-31b-it",
        "openrouter_model": "google/gemma-4-31b-it",
    }


def test_parse_cli_options_parses_dual_provider_custom_model():
    args, option_summary = ehr_input._parse_cli_options(["--openrouter", "--novita", "deepseek/deepseek-vl2", "肺炎"])

    assert args == ["肺炎"]
    assert option_summary == {
        "clear_field": False,
        "dual_provider_mode": True,
        "fireworks_model": None,
        "google_ai_studio": False,
        "novita_model": "deepseek/deepseek-vl2",
        "openrouter_model": "deepseek/deepseek-vl2",
    }


def test_parse_cli_options_rejects_dual_provider_mismatched_models():
    with pytest.raises(RuntimeError, match="同じモデルID"):
        ehr_input._parse_cli_options(["--openrouter", "qwen/qwen3.5-9b", "--novita", "deepseek/deepseek-vl2", "肺炎"])


def test_parse_cli_options_rejects_fireworks_with_openrouter():
    with pytest.raises(RuntimeError, match="同時に指定できません"):
        ehr_input._parse_cli_options(["--fireworks", "accounts/fireworks/models/gemma-4-26b-a4b-it", "--openrouter", "qwen/qwen3.5-9b", "肺炎"])


def test_parse_cli_options_rejects_mactest():
    """--mactest は廃止済み。不明オプションとしてエラーになることを確認。"""
    with pytest.raises(RuntimeError, match="不明なオプション: --mactest"):
        ehr_input._parse_cli_options(["--mactest", "肺炎"])


def test_parse_cli_options_rejects_unknown_option():
    with pytest.raises(RuntimeError, match="不明なオプション: --foobar"):
        ehr_input._parse_cli_options(["--foobar"])


def test_romaji_to_hiragana_len_basic_cases():
    """_romaji_to_hiragana_len がローマ字からひらがな文字数を正しく計算する。"""
    f = ehr_input._romaji_to_hiragana_len
    # 静注: seichuu → せいちゅう (5文字)
    assert f("seichuu") == 5
    # 入院: nyuuin → にゅういん (4文字) — nyu(2) + u(1) + i(1) + n(1)=5? Let me check.
    # nyu → にゅ(2), u → う(1), i → い(1), n(末尾) → ん(1) → total 5
    assert f("nyuuin") == 5
    # 管理: kanri → かんり (3文字)
    assert f("kanri") == 3
    # 病棟: byoutou → びょうとう (5文字)
    assert f("byoutou") == 5
    # 静止: seishi → せいし (3文字)
    assert f("seishi") == 3
    # コントローラー: kontoro-ra- → こんとろーらー (7文字)
    assert f("kontoro-ra-") == 7
    # 促音: kekka → けっか (3文字)
    assert f("kekka") == 3
    # n' handling: kan'i → かんい (3文字)
    assert f("kan'i") == 3
    # 長音ダッシュ: a- → あー (2文字)
    assert f("a-") == 2


def test_validate_vlm_romaji_fixes_particle_ha():
    """助詞「は」のローマ字が wa→ha に修正される。"""
    segments = [
        {"text": "発語", "romaji": "hatsugo"},
        {"text": "は", "romaji": "wa"},
        {"text": "短い", "romaji": "mijikai"},
    ]
    result = ehr_input._validate_vlm_romaji(segments)
    assert result[1] == {"text": "は", "romaji": "ha"}
    assert result[0] == {"text": "発語", "romaji": "hatsugo"}
    assert result[2] == {"text": "短い", "romaji": "mijikai"}


def test_validate_vlm_romaji_fixes_particle_he():
    """助詞「へ」のローマ字が e→he に修正される。"""
    segments = [{"text": "へ", "romaji": "e"}]
    result = ehr_input._validate_vlm_romaji(segments)
    assert result[0] == {"text": "へ", "romaji": "he"}


def test_validate_vlm_romaji_preserves_correct_hiragana_romaji():
    """正しいひらがなローマ字はそのまま保持される。"""
    segments = [{"text": "のみ", "romaji": "nomi"}]
    result = ehr_input._validate_vlm_romaji(segments)
    assert result[0] == {"text": "のみ", "romaji": "nomi"}


def test_validate_vlm_romaji_fixes_hinkai_vowel_drop():
    """頻回 hinka のような末尾母音脱落は cutlet 期待値で補正する。"""
    segments = [{"text": "頻回", "romaji": "hinka"}]
    result = ehr_input._validate_vlm_romaji(segments)
    assert result[0] == {"text": "頻回", "romaji": "hinkai"}


def test_segment_text_for_input_haiya_override_uses_no_for_ya():
    """肺野 override must split into 肺(hai)+野(no), not 野(ya).

    'ya' produces too many candidates (屋,矢,也,野…) making VLM candidate
    number misread likely.  'no' gives a shorter list (の,野,能…) where 野
    appears near position 2, greatly reducing misselection risk.
    """
    result = ehr_input._segment_text_for_input("肺野")
    assert result == [
        {"text": "肺", "romaji": "hai"},
        {"text": "野", "romaji": "no"},
    ]


def test_segment_text_for_input_uses_nichimae_override(monkeypatch):
    monkeypatch.setattr(
        ehr_input,
        "_segment_japanese_with_default_vlm",
        lambda text: [{"text": text, "romaji": "kara"}],
    )
    result = ehr_input._segment_text_for_input("日前から")
    assert result == [
        {"text": "日前", "romaji": "nichimae"},
        {"text": "から", "romaji": "kara"},
    ]


def test_find_best_candidate_match_rejects_shorter_romaji_match(monkeypatch):
    """Romaji pass must reject candidates shorter than target.

    '昭かな' (3 chars, romaji 'akirakana') must NOT match '明らかな'
    (4 chars, romaji 'akirakana') because the shorter candidate is a
    genuinely different word, not an OCR misread.
    """
    monkeypatch.setattr(
        ehr_input,
        "_kanji_to_romaji",
        lambda text: {
            "明らかな": "akirakana",
            "昭かな": "akirakana",
            "昌かな": "akirakana",
        }.get(text, "unknown"),
    )
    candidates = [(2, "昭かな"), (3, "昌かな")]
    result = ehr_input._find_best_candidate_match("明らかな", candidates)
    assert result is None


def test_find_best_candidate_match_allows_longer_romaji_match(monkeypatch):
    """Romaji pass must accept candidates longer-or-equal to target.

    '見とめる' (4 chars) is an OCR expansion of '認める' (3 chars) where
    the first kanji was split into kanji+kana.  Same romaji → accept.
    """
    monkeypatch.setattr(
        ehr_input,
        "_kanji_to_romaji",
        lambda text: {
            "認める": "mitomeru",
            "見とめる": "mitomeru",
        }.get(text, "unknown"),
    )
    candidates = [(5, "見とめる")]
    result = ehr_input._find_best_candidate_match("認める", candidates)
    assert result == (5, "見とめる")


# ------------------------------------------------------------------
# 生食 romaji override (medical: saline = seishoku, not ikezuki)
# ------------------------------------------------------------------

def test_kanji_to_romaji_seishoku():
    """生食 (medical saline) must return 'seishoku', not pykakasi's 'ikezuki'."""
    assert ehr_input._kanji_to_romaji("生食") == "seishoku"


def test_kanji_to_romaji_seichuu():
    """静注 (medical IV injection) must return 'seichuu'."""
    assert ehr_input._kanji_to_romaji("静注") == "seichuu"


def test_kanji_to_romaji_hinkai():
    """頻回 should use cutlet-based reading 'hinkai', not 'hinka'."""
    assert ehr_input._kanji_to_romaji("頻回") == "hinkai"


def test_validate_vlm_romaji_preserves_seishoku():
    """_validate_vlm_romaji must NOT override VLM's correct 'seishoku' for 生食."""
    segments = [{"text": "生食", "romaji": "seishoku"}]
    result = ehr_input._validate_vlm_romaji(segments)
    assert result[0] == {"text": "生食", "romaji": "seishoku"}


# ------------------------------------------------------------------
# kana↔kanji crosstype guard in _ime_candidate_matches
# ------------------------------------------------------------------

def test_ime_candidate_matches_rejects_kana_kanji_crosstype():
    """直地に must NOT match 直ちに (地=kanji ≠ ち=hiragana)."""
    assert ehr_input._ime_candidate_matches("直ちに", "直地に") is False


def test_ime_candidate_matches_accepts_kanji_kanji_noise():
    """著名な must match 著明な (名↔明 are both kanji — legitimate OCR noise, 3+ chars)."""
    assert ehr_input._ime_candidate_matches("著明な", "著名な") is True


def test_ime_candidate_matches_rejects_2char_fuzzy():
    """血競 must NOT match 血症 (2-char fuzzy is disabled — too permissive)."""
    assert ehr_input._ime_candidate_matches("血症", "血競") is False


def test_ime_candidate_matches_2char_exact():
    """血症 exactly matches 血症 even with 2-char length."""
    assert ehr_input._ime_candidate_matches("血症", "血症") is True


# ------------------------------------------------------------------
# Pass 5: visual confusible suffix — kanji-only first chars
# ------------------------------------------------------------------

def test_find_best_candidate_match_rejects_hiragana_first_suffix():
    """Pass 5 must NOT match 'なって' for '伴って' (な=hiragana ≠ 伴=kanji)."""
    candidates = [(1, "なって")]
    result = ehr_input._find_best_candidate_match("伴って", candidates)
    assert result is None


def test_find_best_candidate_match_rejects_kana_suffix_verb():
    """Pass 5 must NOT match '燈って' for '伴って' (suffix 'って' is kana ending)."""
    candidates = [(4, "燈って")]
    result = ehr_input._find_best_candidate_match("伴って", candidates)
    assert result is None


def test_find_best_candidate_match_accepts_kanji_visual_confusible():
    """Pass 5 must match '署明な' for '著明な' (署↔著 are both kanji confusibles)."""
    candidates = [(3, "署明な")]
    result = ehr_input._find_best_candidate_match("著明な", candidates)
    assert result == (3, "署明な")


def test_find_best_candidate_match_rejects_2char_fuzzy_candidate():
    """2-char fuzzy candidate 血競 must NOT match target 血症."""
    candidates = [(4, "血競")]
    result = ehr_input._find_best_candidate_match("血症", candidates)
    assert result is None


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


# ── 分解入力 (Decomposition Typing) テスト ──


def test_decompose_overrides_dict_has_expected_entries():
    """_DECOMPOSE_OVERRIDES に静注と筋注が登録されていることを確認。"""
    assert '静注' in ehr_input._DECOMPOSE_OVERRIDES
    assert '筋注' in ehr_input._DECOMPOSE_OVERRIDES


def test_decompose_overrides_entries_are_valid():
    """各エントリの carrier が keep で始まり、keep が carrier より短いことを確認。"""
    for word, plan in ehr_input._DECOMPOSE_OVERRIDES.items():
        for step in plan:
            assert step['carrier'].startswith(step['keep']), (
                f"{word}: carrier {step['carrier']!r} が keep {step['keep']!r} で始まっていません"
            )
            assert 0 < len(step['keep']) < len(step['carrier']), (
                f"{word}: keep 長が不正 (keep={len(step['keep'])}, carrier={len(step['carrier'])})"
            )


def test_decompose_overrides_disjoint_from_segment_overrides():
    """_DECOMPOSE_OVERRIDES と _SEGMENT_OVERRIDES のキーが重複しないことを確認。"""
    overlap = set(ehr_input._DECOMPOSE_OVERRIDES) & set(ehr_input._SEGMENT_OVERRIDES)
    assert not overlap, f"重複キー: {overlap}"


def test_decompose_seichuu_plan_produces_correct_chars():
    """静注の分解計画が正しい搬送語と残存文字を持つことを確認。"""
    plan = ehr_input._DECOMPOSE_OVERRIDES['静注']
    assert len(plan) == 2
    assert plan[0]['carrier'] == '静脈'
    assert plan[0]['keep'] == '静'
    assert plan[1]['carrier'] == '注射'
    assert plan[1]['keep'] == '注'


def test_type_kanji_via_ime_strict_raises_on_failure(monkeypatch):
    """_strict=True のとき、候補確定できなければ ValueError が発生することを確認。"""
    events = []
    config = SimpleNamespace(
        capture_device_index=0, capture_width=1920, capture_height=1080,
        ocr_languages="ja,en", ocr_use_gpu=False,
    )

    class DummyClient:
        def type_text(self, text):
            events.append(("type", text))
            return True

        def press_key(self, key):
            events.append(("key", key))
            return True

        def send_command(self, cmd):
            events.append(("cmd", cmd))
            return True

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    monkeypatch.setattr(ehr_input, "load_config", lambda skip_password=True: config)
    monkeypatch.setattr(ehr_input, "_wait_for_ble_connected", lambda timeout=70.0: DummyClient())
    monkeypatch.setattr(ehr_input, "capture_screen", lambda **kwargs: dummy_frame)
    monkeypatch.setattr(ehr_input, "_request_ocr_results", lambda *a, **kw: [])
    monkeypatch.setattr(ehr_input, "read_popup_candidates_numbered", lambda *a, **kw: [])
    monkeypatch.setattr(ehr_input, "_read_popup_candidates_with_fallback", lambda *a, **kw: [])
    monkeypatch.setattr(ehr_input, "_read_ime_candidate_with_vlm", lambda *a, **kw: None)
    monkeypatch.setattr(ehr_input, "_save_debug_image", lambda *a, **kw: None)
    monkeypatch.setattr(ehr_input, "detect_ime_mode", lambda *a, **kw: "japanese")

    with pytest.raises(ValueError, match="strict"):
        ehr_input.type_kanji_via_ime(
            "seimyaku", "静脈",
            _current_ime_mode="japanese",
            _strict=True,
        )

    # Verify escape + backspace cleanup happened (no hiragana fallback)
    key_events = [e[1] for e in events if e[0] == "key"]
    assert "right" not in key_events, "strict mode should not use hiragana fallback (right arrow)"


def test_type_kanji_via_decomposition_calls_type_kanji_and_backspace(monkeypatch):
    """分解入力が搬送語ごとに type_kanji_via_ime (strict) + backspace を呼ぶことを確認。"""
    events = []

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

    def fake_type_kanji_via_ime(romaji, target, **kwargs):
        events.append(("ime", romaji, target, kwargs.get("_strict", False)))

    monkeypatch.setattr(ehr_input, "_wait_for_ble_connected", lambda timeout=70.0: DummyClient())
    monkeypatch.setattr(ehr_input, "type_kanji_via_ime", fake_type_kanji_via_ime)

    plan = [
        {'carrier': '静脈', 'romaji': 'seimyaku', 'keep': '静'},
        {'carrier': '注射', 'romaji': 'chuusha', 'keep': '注'},
    ]
    ehr_input._type_kanji_via_decomposition(plan, '静注', 'japanese')

    # Verify: carrier 1 typed in strict mode, then 1 backspace; carrier 2 typed, then 1 backspace
    ime_calls = [(e[1], e[2], e[3]) for e in events if e[0] == "ime"]
    assert ime_calls == [
        ('seimyaku', '静脈', True),
        ('chuusha', '注射', True),
    ]
    backspace_events = [e for e in events if e == ("key", "backspace")]
    assert len(backspace_events) == 2  # 1 for 脈, 1 for 射


def test_type_kanji_via_decomposition_rollback_on_failure(monkeypatch):
    """搬送語の変換失敗時に、既にコミットした文字がロールバックされることを確認。"""
    events = []
    call_count = [0]

    class DummyClient:
        def press_key(self, key):
            events.append(("key", key))
            return True

    def fake_type_kanji_via_ime(romaji, target, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First carrier succeeds
            events.append(("ime_ok", target))
        else:
            # Second carrier fails
            raise ValueError("IME候補に見つかりませんでした（strict モード）")

    monkeypatch.setattr(ehr_input, "_wait_for_ble_connected", lambda timeout=70.0: DummyClient())
    monkeypatch.setattr(ehr_input, "type_kanji_via_ime", fake_type_kanji_via_ime)

    plan = [
        {'carrier': '静脈', 'romaji': 'seimyaku', 'keep': '静'},
        {'carrier': '注射', 'romaji': 'chuusha', 'keep': '注'},
    ]
    with pytest.raises(ValueError, match="ステップ2"):
        ehr_input._type_kanji_via_decomposition(plan, '静注', 'japanese')

    # After step 1 success: 1 backspace for 脈. After step 2 failure: 1 backspace rollback for 静.
    backspace_events = [e for e in events if e == ("key", "backspace")]
    assert len(backspace_events) == 2  # 1 (trim 脈) + 1 (rollback 静)
