import json

import automation.mlx_vlm_ime as mlx_vlm_ime
import automation.mlx_vlm_segmentation as mlx_vlm_segmentation


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_segmentation_uses_runtime_overrides(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["auth"] = req.get_header("Authorization")
        captured["body"] = json.loads(req.data.decode())
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '[{"text":"肺炎","romaji":"haien"}]'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(mlx_vlm_segmentation, "MLX_VLM_SEGMENTATION_URL", "https://openrouter.example/api")
    monkeypatch.setattr(mlx_vlm_segmentation, "MLX_VLM_SEGMENTATION_MODEL", "qwen-test")
    monkeypatch.setattr(mlx_vlm_segmentation, "MLX_VLM_SEGMENTATION_API_KEY", "token-123")
    monkeypatch.setattr(mlx_vlm_segmentation, "MLX_VLM_SEGMENTATION_TIMEOUT", 12.5)
    monkeypatch.setattr(mlx_vlm_segmentation, "MLX_VLM_SEGMENTATION_MAX_TOKENS", 512)
    monkeypatch.setattr(mlx_vlm_segmentation.urllib.request, "urlopen", fake_urlopen)

    _, segments = mlx_vlm_segmentation.segment_japanese_text_with_mlx_vlm("肺炎")

    assert segments == [{"text": "肺炎", "romaji": "haien"}]
    assert captured == {
        "url": "https://openrouter.example/api",
        "timeout": 12.5,
        "auth": "Bearer token-123",
        "body": {
            "model": "qwen-test",
            "messages": [
                {
                    "role": "user",
                    "content": mlx_vlm_segmentation.build_segmentation_prompt("肺炎"),
                }
            ],
            "stream": False,
            "max_tokens": 512,
        },
    }


def test_text_only_ime_call_uses_runtime_overrides(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["auth"] = req.get_header("Authorization")
        captured["body"] = json.loads(req.data.decode())
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "helper response"
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(mlx_vlm_ime, "MLX_VLM_TEXT_URL", "https://openrouter.example/api")
    monkeypatch.setattr(mlx_vlm_ime, "MLX_VLM_TEXT_MODEL", "qwen-text")
    monkeypatch.setattr(mlx_vlm_ime, "MLX_VLM_TEXT_API_KEY", "token-abc")
    monkeypatch.setattr(mlx_vlm_ime, "MLX_VLM_IME_TIMEOUT", 9.0)
    monkeypatch.setattr(mlx_vlm_ime.urllib.request, "urlopen", fake_urlopen)

    content = mlx_vlm_ime._call_mlx_vlm_text_only("helper prompt")

    assert content == "helper response"
    assert captured == {
        "url": "https://openrouter.example/api",
        "timeout": 9.0,
        "auth": "Bearer token-abc",
        "body": {
            "model": "qwen-text",
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": "helper prompt",
                }
            ],
            "stream": False,
            "max_tokens": 512,
        },
    }
