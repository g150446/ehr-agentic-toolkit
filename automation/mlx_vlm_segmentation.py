"""Helpers for probing and parsing mlx_vlm-based Japanese text segmentation."""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request


MLX_VLM_SEGMENTATION_URL = os.getenv(
    "MLX_VLM_SEGMENTATION_URL",
    "http://localhost:8181/v1/chat/completions",
)
MLX_VLM_SEGMENTATION_MODEL = os.getenv(
    "MLX_VLM_SEGMENTATION_MODEL",
    "mlx-community/gemma-4-e2b-it-4bit",
)
MLX_VLM_SEGMENTATION_TIMEOUT = float(os.getenv("MLX_VLM_SEGMENTATION_TIMEOUT", "120"))


class MlxVlmSegmentationError(RuntimeError):
    """Raised when mlx_vlm segmentation fails or returns invalid data."""


def build_segmentation_prompt(text: str) -> str:
    """Build the prompt used to split Japanese text into IME-sized segments."""
    return (
        "以下の日本語文を、WindowsのIMEで1回のEnterで確定できる文節単位のリストに分割してください。\n"
        "各文節について元のテキストとヘボン式ローマ字読みをJSONで出力してください。\n"
        '出力形式（JSONのみ、余分な説明不要）: [{"text": "文節", "romaji": "ローマ字"}]\n\n'
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
        normalized.append({"text": text, "romaji": romaji})

    return normalized


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
