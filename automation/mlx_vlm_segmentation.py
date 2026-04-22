"""Helpers for probing and parsing mlx_vlm-based Japanese text segmentation."""

from __future__ import annotations

import json
import os
import re
import socket
import unicodedata
import urllib.error
import urllib.request
from typing import Optional

from automation.romaji import romanize_for_ime

MLX_VLM_SEGMENTATION_URL = os.getenv(
    "MLX_VLM_SEGMENTATION_URL",
    "http://localhost:8000/v1/chat/completions",
)
MLX_VLM_SEGMENTATION_MODEL = os.getenv(
    "MLX_VLM_SEGMENTATION_MODEL",
    os.getenv("MLX_VLM_SERVER_MODEL", "Qwen3-VL-8B-Instruct-4bit"),
)
MLX_VLM_SEGMENTATION_API_KEY = os.getenv("MLX_VLM_SEGMENTATION_API_KEY", "omlxkey")
MLX_VLM_SEGMENTATION_TIMEOUT = float(os.getenv("MLX_VLM_SEGMENTATION_TIMEOUT", "120"))
MLX_VLM_SEGMENTATION_MAX_TOKENS = int(os.getenv("MLX_VLM_SEGMENTATION_MAX_TOKENS", "512"))
_GOOGLE_AI_STUDIO_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

_SEGMENT_PUNCTUATION_ROMAJI = {
    "、": ",",
    "。": ".",
    "・": "/",
    "（": "(",
    "）": ")",
    "％": "%",
    "：": ":",
    "［": "[",
    "］": "]",
    "【": "[",
    "】": "]",
    "「": "[",
    "」": "]",
    "『": "[",
    "』": "]",
}


class MlxVlmSegmentationError(RuntimeError):
    """Raised when mlx_vlm segmentation fails or returns invalid data."""


def _is_google_ai_studio_url(url: Optional[str]) -> bool:
    return bool(url and url.startswith(_GOOGLE_AI_STUDIO_API_BASE))


def _build_google_ai_studio_request(model: str, prompt: str, url: str) -> tuple[str, dict, dict[str, str]]:
    request_url = f"{url.rstrip('/')}/models/{model}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    return request_url, payload, headers


def _extract_google_ai_studio_text(result: dict) -> str:
    texts: list[str] = []
    for candidate in result.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    if texts:
        return "\n".join(texts).strip()
    raise MlxVlmSegmentationError(
        f"Google AI Studio 応答に content テキストが含まれていません: {result!r}"
    )


def build_segmentation_prompt(text: str) -> str:
    """Build the prompt used to split Japanese text into IME-sized segments."""
    return (
        "以下の日本語文を、WindowsのIMEで1回のEnterで確定できる最小の文節単位に分割してください。\n"
        "ルール:\n"
        "- 助詞（に、で、が、を、は、も、へ、から、まで、と、の、より、や）は必ず独立した文節にする\n"
        "- 日本語の文節だけを返し、ASCII記号や改行は分割対象に含めない\n"
        "- 1文節は最大4〜5文字程度にする\n"
        '出力形式（JSONのみ、余分な説明・コードブロック不要）: [{"text": "文節"}]\n'
        "- romaji や読み方は出力しない\n\n"
        f"入力: {text}"
    )


def _repair_json_array(raw: str) -> str:
    """Fix common malformed JSON patterns from VLM output.

    Handles cases like: [{"text": "x", "romaji": "y"], {"text": ...}]
    where a closing `]` appears instead of `},` between objects.
    """
    # Replace `], {` (missing closing brace) with `}, {`
    raw = re.sub(r'"\s*\]\s*,\s*\{', '"}, {', raw)
    # Replace `"x"] }` style endings within an object
    raw = re.sub(r'"\s*\]\s*\}', '"}', raw)
    return raw


def parse_segment_response(content: str) -> list[dict[str, str]]:
    """Extract segment texts from VLM output and compute romaji locally."""
    # Strip <think>...</think> blocks (Qwen3 thinking output)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    start = content.find("[")
    end = content.rfind("]") + 1
    if start == -1 or end == 0:
        raise MlxVlmSegmentationError(
            f"mlx_vlm応答からJSON配列を抽出できませんでした: {content!r}"
        )

    raw = content[start:end]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Try repairing common malformed patterns
        repaired = _repair_json_array(raw)
        try:
            payload = json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise MlxVlmSegmentationError(
                f"mlx_vlm応答のJSON配列を解析できませんでした: {content!r}"
            ) from exc

    if not isinstance(payload, list):
        raise MlxVlmSegmentationError(
            f"mlx_vlm応答のJSON配列が不正です: {payload!r}"
        )

    normalized: list[dict[str, str]] = []
    for index, item in enumerate(payload, start=1):
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = item.get("text")
        else:
            raise MlxVlmSegmentationError(
                f"mlx_vlm応答の要素 {index} が文字列またはオブジェクトではありません: {item!r}"
            )
        if not isinstance(text, str) or not text:
            raise MlxVlmSegmentationError(
                f"mlx_vlm応答の要素 {index} に有効な text がありません: {item!r}"
            )
        if text in _SEGMENT_PUNCTUATION_ROMAJI:
            normalized_romaji = _SEGMENT_PUNCTUATION_ROMAJI[text]
        elif text.isascii():
            normalized_romaji = text
        else:
            normalized_romaji = _normalize_romaji(romanize_for_ime(text))
        if not normalized_romaji:
            raise MlxVlmSegmentationError(
                f"mlx_vlm応答の要素 {index} のローマ字をローカル生成できませんでした: {item!r}"
            )
        normalized.append({"text": text, "romaji": normalized_romaji})

    return normalized


def _normalize_romaji(romaji: str) -> str:
    """ローマ字を IME 入力可能な ASCII 文字列に正規化する。

    - スペースを除去
    - 長音符（ō ū ā等）を ASCII に展開
    - NFKDで合成文字を分解してから非ASCII文字を除去
    """
    # 長音符の明示的な置換（NFKDより先に行う）
    _long_vowel_map = str.maketrans({
        "ā": "a", "Ā": "A",
        "ī": "i", "Ī": "I",
        "ū": "u", "Ū": "U",
        "ē": "e", "Ē": "E",
        "ō": "o", "Ō": "O",
    })
    romaji = romaji.translate(_long_vowel_map)
    # NFKDで分解して残った結合文字（アクセント等）を除去
    romaji = "".join(
        ch for ch in unicodedata.normalize("NFKD", romaji)
        if unicodedata.category(ch) != "Mn"
    )
    # スペースと記号を除去、小文字化
    romaji = re.sub(r"[^a-zA-Z]", "", romaji).lower()
    return romaji


def segment_japanese_text_with_mlx_vlm(
    text: str,
    *,
    model: Optional[str] = None,
    url: Optional[str] = None,
    timeout: Optional[float] = None,
    api_key: Optional[str] = None,
) -> tuple[str, list[dict[str, str]]]:
    """Call local mlx_vlm server and return both raw text output and parsed segments."""
    model = model or MLX_VLM_SEGMENTATION_MODEL
    url = url or MLX_VLM_SEGMENTATION_URL
    timeout = MLX_VLM_SEGMENTATION_TIMEOUT if timeout is None else timeout
    api_key = api_key or MLX_VLM_SEGMENTATION_API_KEY

    prompt = build_segmentation_prompt(text)
    request_url = url
    headers = {"Content-Type": "application/json"}
    if _is_google_ai_studio_url(url):
        request_url, payload, headers = _build_google_ai_studio_request(model, prompt, url)
        headers["x-goog-api-key"] = api_key
    else:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": MLX_VLM_SEGMENTATION_MAX_TOKENS,
        }
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        request_url,
        data=json.dumps(payload).encode(),
        headers=headers,
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            response_body = resp.read()
    except (TimeoutError, socket.timeout) as exc:
        raise MlxVlmSegmentationError(
            f"mlx_vlmへのリクエストが {timeout:g} 秒でタイムアウトしました。"
            f" endpoint={url} model={model}"
        ) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise MlxVlmSegmentationError(
                f"mlx_vlmへのリクエストが {timeout:g} 秒でタイムアウトしました。"
                f" endpoint={url} model={model}"
            ) from exc
        raise MlxVlmSegmentationError(
            f"mlx_vlmへの接続に失敗しました: {reason}. endpoint={url} model={model}"
        ) from exc

    try:
        result = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise MlxVlmSegmentationError(
            f"mlx_vlm応答のJSONを解析できませんでした: {response_body!r}"
        ) from exc

    try:
        if _is_google_ai_studio_url(url):
            content = _extract_google_ai_studio_text(result)
        else:
            content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise MlxVlmSegmentationError(
            f"mlx_vlm応答に content テキストが含まれていません: {result!r}"
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise MlxVlmSegmentationError(
            f"mlx_vlm応答に content テキストが含まれていません: {result!r}"
        )

    return content, parse_segment_response(content)
