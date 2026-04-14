"""Helpers for probing and parsing mlx_vlm-based Japanese text segmentation."""

from __future__ import annotations

import json
import os
import re
import socket
import unicodedata
import urllib.error
import urllib.request


MLX_VLM_SEGMENTATION_URL = os.getenv(
    "MLX_VLM_SEGMENTATION_URL",
    "http://localhost:8181/v1/chat/completions",
)
MLX_VLM_SEGMENTATION_MODEL = os.getenv(
    "MLX_VLM_SEGMENTATION_MODEL",
    os.getenv("MLX_VLM_SERVER_MODEL", "mlx-community/Qwen3.5-4B-MLX-4bit"),
)
MLX_VLM_SEGMENTATION_TIMEOUT = float(os.getenv("MLX_VLM_SEGMENTATION_TIMEOUT", "120"))


class MlxVlmSegmentationError(RuntimeError):
    """Raised when mlx_vlm segmentation fails or returns invalid data."""


def build_segmentation_prompt(text: str) -> str:
    """Build the prompt used to split Japanese text into IME-sized segments."""
    return (
        "以下の日本語文を、WindowsのIMEで1回のEnterで確定できる最小の文節単位に分割してください。\n"
        "ルール:\n"
        "- 助詞（に、で、が、を、は、も、へ、から、まで、と、の、より、や）は必ず独立した文節にする\n"
        "- 日本語の文節だけを返し、ASCII記号や改行は分割対象に含めない\n"
        "- romaji はスペースなしのヘボン式（長音は重ねる: おう→ou, こう→ko u）\n"
        "- 1文節は最大4〜5文字程度にする\n"
        '出力形式（JSONのみ、余分な説明・コードブロック不要）: [{"text": "文節", "romaji": "ローマ字"}]\n\n'
        f"入力: {text}"
    )


def parse_segment_response(content: str) -> list[dict[str, str]]:
    """Extract and validate the JSON segment array from mlx_vlm text output."""
    start = content.find("[")
    end = content.rfind("]") + 1
    if start == -1 or end == 0:
        raise MlxVlmSegmentationError(
            f"mlx_vlm応答からJSON配列を抽出できませんでした: {content!r}"
        )

    try:
        payload = json.loads(content[start:end])
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
        if not isinstance(item, dict):
            raise MlxVlmSegmentationError(
                f"mlx_vlm応答の要素 {index} がオブジェクトではありません: {item!r}"
            )
        text = item.get("text")
        romaji = item.get("romaji")
        if not isinstance(text, str) or not text:
            raise MlxVlmSegmentationError(
                f"mlx_vlm応答の要素 {index} に有効な text がありません: {item!r}"
            )
        if not isinstance(romaji, str) or not romaji:
            raise MlxVlmSegmentationError(
                f"mlx_vlm応答の要素 {index} に有効な romaji がありません: {item!r}"
            )
        normalized.append({"text": text, "romaji": _normalize_romaji(romaji)})

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
    model: str = MLX_VLM_SEGMENTATION_MODEL,
    url: str = MLX_VLM_SEGMENTATION_URL,
    timeout: float = MLX_VLM_SEGMENTATION_TIMEOUT,
) -> tuple[str, list[dict[str, str]]]:
    """Call local mlx_vlm server and return both raw text output and parsed segments."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": build_segmentation_prompt(text)}],
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
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
