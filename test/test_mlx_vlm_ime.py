from contextlib import redirect_stdout
import io

import automation.mlx_vlm_ime as mlx_vlm_ime


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


def test_read_popup_candidates_numbered_vlm_parses_natural_language_reasoning(monkeypatch):
    monkeypatch.setattr(mlx_vlm_ime, "_crop_popup_region", lambda frame, debug_name="": frame)
    monkeypatch.setattr(mlx_vlm_ime, "_encode_image_data_url", lambda frame: "data:image/png;base64,xxx")
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
    monkeypatch.setattr(mlx_vlm_ime, "_encode_image_data_url", lambda frame: "data:image/png;base64,xxx")
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
