from contextlib import redirect_stdout
import io
import json
import sys
import types

import automation.mlx_vlm_ime as mlx_vlm_ime
import numpy as np


def test_extract_message_text_prefers_content():
    result = {
        "choices": [
            {
                "message": {
                    "content": "english",
                    "reasoning": "japanese",
                }
            }
        ]
    }

    assert mlx_vlm_ime._extract_message_text(result, error_prefix="mlx_vlm IME") == "english"


def test_extract_message_text_falls_back_to_reasoning():
    result = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "reasoning": "japanese",
                    "reasoning_details": [
                        {"text": "japanese"},
                    ],
                }
            }
        ]
    }

    assert mlx_vlm_ime._extract_message_text(result, error_prefix="mlx_vlm IME") == "japanese"


def test_extract_message_text_handles_real_log_style_reasoning_only_response():
    result = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "reasoning": "用户希望我识别图像中新增的字符。输出 'japanese'",
                    "reasoning_details": [
                        {
                            "type": "reasoning.text",
                            "text": "用户希望我识别图像中新增的字符。输出 'japanese'",
                        }
                    ],
                }
            }
        ]
    }

    assert "japanese" in mlx_vlm_ime._extract_message_text(result, error_prefix="mlx_vlm IME")


def test_extract_diff_crop_stores_active_typing_line_hint():
    pre = np.zeros((120, 160, 3), dtype=np.uint8)
    post = pre.copy()
    post[70:86, 20:34] = 255

    mlx_vlm_ime.reset_active_typing_line_hint()

    cropped = mlx_vlm_ime._extract_diff_crop(pre, post, pad=0)

    hint = mlx_vlm_ime.get_active_typing_line_hint()
    assert cropped is not None
    assert hint is not None
    assert 0.5 < hint["center_y_ratio"] < 0.8
    assert hint["char_height_ratio"] > 0


def test_crop_to_active_typing_line_uses_normalized_hint():
    frame = np.zeros((100, 80, 3), dtype=np.uint8)

    cropped = mlx_vlm_ime.crop_to_active_typing_line(
        frame,
        {"center_y_ratio": 0.7, "char_height_ratio": 0.1},
    )

    assert cropped is not None
    assert 30 <= cropped.shape[0] <= 60


def test_resolve_paddle_model_dir_prefers_paddlex_cache(tmp_path, monkeypatch):
    home = tmp_path / "home"
    model_dir = home / ".paddlex" / "official_models" / "PP-DocLayout_plus-L"
    model_dir.mkdir(parents=True)
    (model_dir / "inference.yml").write_text("test", encoding="utf-8")

    monkeypatch.setattr(mlx_vlm_ime.Path, "home", classmethod(lambda cls: home))

    assert mlx_vlm_ime._resolve_paddle_model_dir("PP-DocLayout_plus-L") == str(model_dir)


def test_load_ppstructure_popup_engine_uses_local_model_dirs(monkeypatch):
    captured = {}

    class DummyPPStructure:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PPStructureV3=DummyPPStructure))
    monkeypatch.setattr(
        mlx_vlm_ime,
        "_resolve_paddle_model_dir",
        lambda name: f"/tmp/{name}",
    )
    monkeypatch.setattr(mlx_vlm_ime, "_ppstructure_popup_engine", None)
    monkeypatch.setattr(mlx_vlm_ime, "_ppstructure_popup_engine_failed", False)

    engine = mlx_vlm_ime._load_ppstructure_popup_engine()

    assert isinstance(engine, DummyPPStructure)
    assert captured["use_region_detection"] is False
    assert captured["layout_detection_model_dir"] == "/tmp/PP-DocLayout_plus-L"
    assert captured["text_detection_model_dir"] == "/tmp/PP-OCRv5_server_det"
    assert captured["text_recognition_model_dir"] == "/tmp/PP-OCRv5_server_rec"


def test_read_popup_candidates_numbered_vlm_parses_natural_language_reasoning(monkeypatch):
    monkeypatch.setattr(mlx_vlm_ime, "_crop_popup_region", lambda frame, debug_name="": frame)
    monkeypatch.setattr(mlx_vlm_ime, "_encode_image_data_url", lambda frame, **kwargs: "data:image/png;base64,xxx")
    monkeypatch.setattr(
        mlx_vlm_ime,
        "_call_mlx_vlm_with_image",
        lambda data_url, prompt: (
            "Item 1: The number is `1` and the text next to it is `過`.\n"
            "Item 2: The number is `2` and the text next to it is `膨張`."
        ),
    )

    assert mlx_vlm_ime.read_popup_candidates_numbered_vlm(object()) == [(1, "過"), (2, "膨張")]


def test_read_popup_candidates_numbered_vlm_parses_real_log_style_item_lines(monkeypatch):
    monkeypatch.setattr(mlx_vlm_ime, "_crop_popup_region", lambda frame, debug_name="": frame)
    monkeypatch.setattr(mlx_vlm_ime, "_encode_image_data_url", lambda frame, **kwargs: "data:image/png;base64,xxx")
    monkeypatch.setattr(
        mlx_vlm_ime,
        "_call_mlx_vlm_with_image",
        lambda data_url, prompt: (
            "Based on the visual content of the image, I need to extract the numbered list items from the IME popup.\n\n"
            "1.  **Item 1:** The number is `1` and the text next to it is `テスト`.\n"
            "2.  **Item 2:** The number is `2` and the text next to it is `test`.\n"
            "3.  **Item 3:** The number is `3` and the text next to it is `t e s t`."
        ),
    )

    assert mlx_vlm_ime.read_popup_candidates_numbered_vlm(object()) == [
        (1, "テスト"),
        (2, "test"),
        (3, "t e s t"),
    ]


def test_suggest_ime_helper_word_logs_openrouter_runtime(monkeypatch):
    stdout = io.StringIO()

    monkeypatch.setattr(mlx_vlm_ime, "MLX_VLM_TEXT_URL", "https://openrouter.ai/api/v1/chat/completions")
    monkeypatch.setattr(mlx_vlm_ime, "MLX_VLM_TEXT_MODEL", "google/gemma-4-26b-a4b-it")
    monkeypatch.setattr(
        mlx_vlm_ime,
        "_call_mlx_vlm_text_only",
        lambda prompt, model=None: '{"words":["過剰","過去","過失"]}',
    )

    with redirect_stdout(stdout):
        assert mlx_vlm_ime.suggest_ime_helper_word("過") == [
            {"word": "過剰", "backspace_count": 1},
            {"word": "過去", "backspace_count": 1},
            {"word": "過失", "backspace_count": 1},
        ]

    output = stdout.getvalue()
    assert "OpenRouter(google/gemma-4-26b-a4b-it)応答" in output
    assert "Qwen3応答" not in output


def test_suggest_ime_helper_word_logs_google_ai_studio_runtime(monkeypatch):
    stdout = io.StringIO()

    monkeypatch.setattr(mlx_vlm_ime, "MLX_VLM_TEXT_URL", "https://generativelanguage.googleapis.com/v1beta")
    monkeypatch.setattr(mlx_vlm_ime, "MLX_VLM_TEXT_MODEL", "gemma-4-26b-a4b-it")
    monkeypatch.setattr(
        mlx_vlm_ime,
        "_call_mlx_vlm_text_only",
        lambda prompt, model=None: '{"words":["過剰","過去","過失"]}',
    )

    with redirect_stdout(stdout):
        assert mlx_vlm_ime.suggest_ime_helper_word("過") == [
            {"word": "過剰", "backspace_count": 1},
            {"word": "過去", "backspace_count": 1},
            {"word": "過失", "backspace_count": 1},
        ]

    output = stdout.getvalue()
    assert "Google AI Studio(gemma-4-26b-a4b-it)応答" in output
    assert "Qwen3応答" not in output


def test_suggest_ime_helper_word_logs_novita_runtime(monkeypatch):
    stdout = io.StringIO()

    monkeypatch.setattr(mlx_vlm_ime, "MLX_VLM_TEXT_URL", "https://api.novita.ai/openai/chat/completions")
    monkeypatch.setattr(mlx_vlm_ime, "MLX_VLM_TEXT_MODEL", "google/gemma-4-31b-it")
    monkeypatch.setattr(
        mlx_vlm_ime,
        "_call_mlx_vlm_text_only",
        lambda prompt, model=None: '{"words":["過剰","過去","過失"]}',
    )

    with redirect_stdout(stdout):
        assert mlx_vlm_ime.suggest_ime_helper_word("過") == [
            {"word": "過剰", "backspace_count": 1},
            {"word": "過去", "backspace_count": 1},
            {"word": "過失", "backspace_count": 1},
        ]

    output = stdout.getvalue()
    assert "Novita(google/gemma-4-31b-it)応答" in output
    assert "Qwen3応答" not in output


def test_classify_helper_reset_screen_keeps_patient_record_without_vlm(monkeypatch):
    frame = np.zeros((120, 240, 3), dtype=np.uint8)

    monkeypatch.setattr(mlx_vlm_ime, "detect_patient_record_panel3", lambda image: (40, 160))
    monkeypatch.setattr(
        mlx_vlm_ime,
        "_call_mlx_vlm_with_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("VLM should not be called")),
    )

    assert mlx_vlm_ime.classify_helper_reset_screen(frame) == "patient_record"


def test_prime_helper_reset_panel_cache_validates_and_reuses_bounds(monkeypatch):
    frame = np.zeros((120, 240, 3), dtype=np.uint8)

    mlx_vlm_ime.reset_helper_reset_panel_cache()
    monkeypatch.setattr(mlx_vlm_ime, "detect_patient_record_panel3", lambda image, debug_name="": (40, 160))
    monkeypatch.setattr(mlx_vlm_ime, "_encode_image_data_url", lambda image, **kwargs: "data:image/mock")
    monkeypatch.setattr(mlx_vlm_ime, "_call_mlx_vlm_with_images", lambda *args, **kwargs: "yes")

    assert mlx_vlm_ime.prime_helper_reset_panel_cache(frame, debug_name="helper_reset_initial") == (40, 160)
    assert mlx_vlm_ime.get_helper_reset_panel_cache() == (40, 160)

    monkeypatch.setattr(
        mlx_vlm_ime,
        "detect_patient_record_panel3",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("panel detection should not rerun")),
    )

    cropped, screen_type = mlx_vlm_ime.crop_helper_reset_region(
        frame,
        screen_type="patient_record",
        debug_name="helper_reset",
    )

    assert screen_type == "patient_record"
    assert cropped.shape == (120, 120, 3)
    mlx_vlm_ime.reset_helper_reset_panel_cache()


def test_crop_notepad_document_region_excludes_taskbar_and_menu():
    frame = np.full((300, 400, 3), 180, dtype=np.uint8)
    frame[20:40, 50:350] = 210
    frame[40:240, 50:350] = 255
    frame[270:, :] = 20

    cropped = mlx_vlm_ime.crop_notepad_document_region(frame)

    assert cropped.shape == (200, 300, 3)
    assert np.all(cropped == 255)


def test_crop_notepad_document_region_trims_white_window_header():
    frame = np.full((1080, 1920, 3), 255, dtype=np.uint8)
    frame[0:30, :] = 40
    frame[30:105, :] = 230
    frame[1005:1080, :] = 225
    frame[1006:1013, :] = 30
    frame[:, 1910:] = 235

    cropped = mlx_vlm_ime.crop_notepad_document_region(frame)

    assert cropped.shape[0] < 1000
    assert cropped.shape[1] < 1920
    assert cropped.shape[0] > 700
    assert cropped.shape[1] > 1500


def test_crop_helper_reset_region_uses_notepad_branch(monkeypatch):
    frame = np.zeros((120, 240, 3), dtype=np.uint8)

    monkeypatch.setattr(mlx_vlm_ime, "classify_helper_reset_screen", lambda image, debug_name="": "notepad")
    monkeypatch.setattr(
        mlx_vlm_ime,
        "crop_notepad_document_region",
        lambda image, debug_name="": np.full((40, 80, 3), 255, dtype=np.uint8),
    )

    cropped, screen_type = mlx_vlm_ime.crop_helper_reset_region(frame, debug_name="helper_reset")

    assert screen_type == "notepad"
    assert cropped.shape == (40, 80, 3)


def test_crop_popup_region_limits_patient_record_to_panel_before_blue_detection(monkeypatch):
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    calls = []

    monkeypatch.setattr(mlx_vlm_ime, "detect_patient_record_panel3", lambda image: (40, 160))
    monkeypatch.setattr(
        mlx_vlm_ime,
        "crop_to_input_region",
        lambda image, debug_name="", panel_bounds=None: calls.append(
            ("panel", image.shape, debug_name, panel_bounds)
        ) or np.ones((90, 120, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        mlx_vlm_ime,
        "_crop_center_band",
        lambda image, top_ratio=0.25, bottom_ratio=0.85: calls.append(
            ("band", image.shape, top_ratio, bottom_ratio)
        ) or np.ones((54, 120, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        mlx_vlm_ime,
        "crop_to_ime_popup_by_blue",
        lambda image: calls.append(("blue", image.shape)) or np.full((30, 50, 3), 255, dtype=np.uint8),
    )

    cropped = mlx_vlm_ime._crop_popup_region(frame, debug_name="line")

    assert cropped.shape == (30, 50, 3)
    assert calls == [
        ("panel", frame.shape, "popup_region_line", (40, 160)),
        ("band", (90, 120, 3), 0.25, 0.85),
        ("blue", (54, 120, 3)),
    ]


def test_crop_popup_region_uses_cyan_row_extractor_before_fallback(monkeypatch):
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    calls = []
    cyan_crop = np.full((24, 48, 3), 200, dtype=np.uint8)

    monkeypatch.setattr(mlx_vlm_ime, "detect_patient_record_panel3", lambda image: (40, 160))
    monkeypatch.setattr(
        mlx_vlm_ime,
        "crop_to_input_region",
        lambda image, debug_name="", panel_bounds=None: np.ones((90, 120, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        mlx_vlm_ime,
        "_crop_center_band",
        lambda image, top_ratio=0.25, bottom_ratio=0.85: np.ones((54, 120, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        mlx_vlm_ime,
        "crop_to_ime_popup_by_blue",
        lambda image: calls.append(("blue", image.shape)) or cyan_crop,
    )

    cropped = mlx_vlm_ime._crop_popup_region(frame, debug_name="line")

    assert cropped is cyan_crop
    assert calls == [("blue", (54, 120, 3))]


def test_crop_popup_region_uses_full_frame_blue_detection_when_not_patient_record(monkeypatch):
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    calls = []

    monkeypatch.setattr(mlx_vlm_ime, "detect_patient_record_panel3", lambda image: None)
    monkeypatch.setattr(
        mlx_vlm_ime,
        "crop_to_ime_popup_by_blue",
        lambda image: calls.append(("blue", image.shape)) or np.full((20, 40, 3), 255, dtype=np.uint8),
    )
    monkeypatch.setattr(
        mlx_vlm_ime,
        "crop_to_input_region",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fallback should not run")),
    )

    cropped = mlx_vlm_ime._crop_popup_region(frame, debug_name="line")

    assert cropped.shape == (20, 40, 3)
    assert calls == [("blue", frame.shape)]


def test_crop_to_ime_popup_by_blue_extracts_popup_from_cyan_selected_row():
    frame = np.zeros((320, 240, 3), dtype=np.uint8)
    frame[90:112, 60:220] = np.array([255, 255, 0], dtype=np.uint8)

    cropped = mlx_vlm_ime.crop_to_ime_popup_by_blue(frame)

    assert cropped is not None
    assert cropped.shape[1] == 170
    assert cropped.shape[0] == int(22 * 9.5)


def test_crop_popup_region_by_ppstructure_selects_table_like_block(monkeypatch):
    frame = np.zeros((120, 240, 3), dtype=np.uint8)

    class DummyEngine:
        def predict(self, image, **kwargs):
            return [
                {
                    "layout_det_res": [
                        {"label": "text", "bbox": [5, 5, 50, 25]},
                        {"label": "table", "bbox": [30, 40, 160, 110]},
                    ]
                }
            ]

    monkeypatch.setattr(mlx_vlm_ime, "_load_ppstructure_popup_engine", lambda: DummyEngine())

    cropped = mlx_vlm_ime._crop_popup_region_by_ppstructure(frame, debug_name="table")

    assert cropped is not None
    assert cropped.shape == (70, 130, 3)


def test_read_popup_candidates_ocr_uses_popup_region_extractor(monkeypatch):
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    popup_crop = np.ones((80, 120, 3), dtype=np.uint8)

    monkeypatch.setattr(mlx_vlm_ime, "_crop_popup_region", lambda image, debug_name="": popup_crop)
    import automation.screen_analyzer as screen_analyzer

    monkeypatch.setattr(screen_analyzer, "load_ocr_reader", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        screen_analyzer,
        "run_ocr",
        lambda reader, image: [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "1", 0.99),
            ([[20, 0], [60, 0], [60, 10], [20, 10]], "過剰", 0.99),
        ],
    )

    assert mlx_vlm_ime.read_popup_candidates_ocr(frame, debug_name="popup") == [(1, "過剰")]


def test_save_thinking_log_writes_reasoning_details(tmp_path, monkeypatch):
    monkeypatch.setattr(mlx_vlm_ime, "_logs_dir", lambda: str(tmp_path))

    mlx_vlm_ime._save_thinking_log(
        prompt="compare prompt",
        image_count=2,
        model="test-model",
        url="http://localhost:8000",
        reasoning_requested=True,
        result={
            "choices": [
                {
                    "message": {
                        "content": "yes",
                        "reasoning": "main reasoning",
                        "reasoning_details": [{"text": "detail one"}],
                    }
                }
            ]
        },
    )

    logs = list(tmp_path.glob("*_thinking.txt"))
    assert len(logs) == 1
    text = logs[0].read_text(encoding="utf-8")
    assert "image_count: 2" in text
    assert "reasoning_requested: True" in text
    assert "reasoning_present: True" in text
    assert "compare prompt" in text
    assert "main reasoning" in text
    assert "detail one" in text


def test_wait_for_vlm_cooldown_sleeps_until_half_second(monkeypatch):
    events = []

    monkeypatch.setattr(mlx_vlm_ime, "_last_vlm_response_monotonic", 10.0)
    monkeypatch.setattr(mlx_vlm_ime.time, "monotonic", lambda: 10.25)
    monkeypatch.setattr(mlx_vlm_ime.time, "sleep", lambda seconds: events.append(seconds))

    mlx_vlm_ime._wait_for_vlm_cooldown()

    assert events == [0.25]


def test_call_mlx_vlm_with_content_uses_openrouter_provider_for_reasoning(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"yes","reasoning":"trace"}}]}'

    def _fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        captured["headers"] = dict(req.header_items())
        return FakeResponse()

    monkeypatch.setattr(mlx_vlm_ime, "_wait_for_vlm_cooldown", lambda: None)
    monkeypatch.setattr(mlx_vlm_ime.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(mlx_vlm_ime.time, "monotonic", lambda: 10.0)

    result = mlx_vlm_ime._call_mlx_vlm_with_content(
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}}],
        "compare prompt",
        url="https://openrouter.ai/api/v1/chat/completions",
        api_key="token",
        enable_reasoning=True,
    )

    assert result == "yes"
    assert captured["payload"]["include_reasoning"] is True
    assert captured["payload"]["provider"] == {"order": ["io-net"]}
    assert "reasoning" not in captured["payload"]
    assert captured["headers"]["Http-referer"] == "https://github.com/g150446/ehr-agentic-toolkit"
    assert captured["headers"]["X-title"] == "EHR Agentic Toolkit"


def test_call_mlx_vlm_with_content_uses_google_ai_studio_payload(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"candidates":[{"content":{"parts":[{"text":"yes"}]}}]}'

    def _fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        captured["headers"] = dict(req.header_items())
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(mlx_vlm_ime, "_wait_for_vlm_cooldown", lambda: None)
    monkeypatch.setattr(mlx_vlm_ime.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(mlx_vlm_ime.time, "monotonic", lambda: 10.0)

    result = mlx_vlm_ime._call_mlx_vlm_with_content(
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}}],
        "compare prompt",
        model="gemma-4-26b-a4b-it",
        url="https://generativelanguage.googleapis.com/v1beta",
        api_key="token",
        enable_reasoning=True,
    )

    assert result == "yes"
    assert captured["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent"
    assert captured["timeout"] == mlx_vlm_ime.MLX_VLM_IME_TIMEOUT
    assert captured["headers"]["X-goog-api-key"] == "token"
    assert captured["payload"] == {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "compare prompt"},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": "xxx",
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "thinkingConfig": {
                "thinkingLevel": "high",
            }
        },
    }


def test_call_mlx_vlm_with_content_uses_fireworks_openai_payload(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"yes","reasoning":"trace"}}]}'

    def _fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        captured["headers"] = dict(req.header_items())
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(mlx_vlm_ime, "_wait_for_vlm_cooldown", lambda: None)
    monkeypatch.setattr(mlx_vlm_ime.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(mlx_vlm_ime.time, "monotonic", lambda: 10.0)

    result = mlx_vlm_ime._call_mlx_vlm_with_content(
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}}],
        "compare prompt",
        model="accounts/fireworks/models/gemma-4-26b-a4b-it",
        url="https://api.fireworks.ai/inference/v1/chat/completions",
        api_key="token",
        enable_reasoning=True,
    )

    assert result == "yes"
    assert captured["url"] == "https://api.fireworks.ai/inference/v1/chat/completions"
    assert captured["timeout"] == mlx_vlm_ime.MLX_VLM_IME_TIMEOUT
    assert captured["headers"] == {
        "Authorization": "Bearer token",
        "Content-type": "application/json",
    }
    assert captured["payload"]["model"] == "accounts/fireworks/models/gemma-4-26b-a4b-it"
    assert captured["payload"]["include_reasoning"] is True
    assert captured["payload"]["reasoning"] == {"enabled": True}
    assert "provider" not in captured["payload"]


def test_call_mlx_vlm_with_content_uses_novita_openai_client(monkeypatch):
    captured = {}

    class FakeChatCompletions:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs

            class FakeResponse:
                def model_dump(self_inner):
                    return {"choices": [{"message": {"content": "yes"}}]}

            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = type("ChatNamespace", (), {"completions": FakeChatCompletions()})()

    monkeypatch.setattr(mlx_vlm_ime, "_wait_for_vlm_cooldown", lambda: None)
    monkeypatch.setattr(mlx_vlm_ime, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(mlx_vlm_ime.time, "monotonic", lambda: 10.0)

    result = mlx_vlm_ime._call_mlx_vlm_with_content(
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}}],
        "compare prompt",
        model="google/gemma-4-31b-it",
        url="https://api.novita.ai/openai/chat/completions",
        api_key="token",
        enable_reasoning=True,
    )

    assert result == "yes"
    assert captured["client_kwargs"] == {
        "api_key": "token",
        "base_url": "https://api.novita.ai/openai",
        "timeout": mlx_vlm_ime.MLX_VLM_IME_TIMEOUT,
        "max_retries": 0,
    }
    assert captured["kwargs"] == {
        "model": "google/gemma-4-31b-it",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "compare prompt"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
                ],
            }
        ],
        "stream": False,
        "max_tokens": 256,
        "temperature": 0,
    }


def test_suggest_ime_helper_word_logs_fireworks_runtime(monkeypatch):
    stdout = io.StringIO()

    monkeypatch.setattr(mlx_vlm_ime, "MLX_VLM_TEXT_URL", "https://api.fireworks.ai/inference/v1/chat/completions")
    monkeypatch.setattr(mlx_vlm_ime, "MLX_VLM_TEXT_MODEL", "accounts/fireworks/models/gemma-4-26b-a4b-it")
    monkeypatch.setattr(
        mlx_vlm_ime,
        "_call_mlx_vlm_text_only",
        lambda prompt, model=None: '{"words":["過剰","過去","過失"]}',
    )

    with redirect_stdout(stdout):
        assert mlx_vlm_ime.suggest_ime_helper_word("過") == [
            {"word": "過剰", "backspace_count": 1},
            {"word": "過去", "backspace_count": 1},
            {"word": "過失", "backspace_count": 1},
        ]

    output = stdout.getvalue()
    assert "Fireworks(accounts/fireworks/models/gemma-4-26b-a4b-it)応答" in output
    assert "Qwen3応答" not in output
