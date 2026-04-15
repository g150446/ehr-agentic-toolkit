"""mlx_vlm server を使った IME 候補読み取りヘルパー。

mlx_vlm.server（OpenAI 互換 API）に画像を送信し、
IME インライン変換候補やポップアップ候補リストを読み取る。
"""

from __future__ import annotations

import base64
import json
import os
import re
import socket
import urllib.error
import urllib.request
from typing import Optional

import cv2
import numpy as np

MLX_VLM_IME_URL = os.getenv(
    "MLX_VLM_IME_URL",
    os.getenv("MLX_VLM_SEGMENTATION_URL", "http://localhost:8181/v1/chat/completions"),
)
MLX_VLM_IME_MODEL = os.getenv(
    "MLX_VLM_IME_MODEL",
    os.getenv("MLX_VLM_SERVER_MODEL", "mlx-community/Qwen3-VL-8B-Instruct-4bit"),
)
MLX_VLM_IME_TIMEOUT = float(os.getenv("MLX_VLM_IME_TIMEOUT", "90"))


class MlxVlmImeError(RuntimeError):
    """Raised when mlx_vlm IME call fails."""


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

_MIN_FRAME_HEIGHT = 80
_MIN_FRAME_WIDTH = 200


def _ensure_min_size(frame: np.ndarray) -> np.ndarray:
    """VLM が処理できる最小サイズに拡大する（アスペクト比維持）。"""
    h, w = frame.shape[:2]
    scale = max(_MIN_FRAME_HEIGHT / h, _MIN_FRAME_WIDTH / w, 1.0)
    if scale <= 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def _encode_image_data_url(frame: np.ndarray) -> str:
    """numpy BGR フレームを data URI 形式の PNG base64 文字列に変換する。"""
    frame = _ensure_min_size(frame)
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise MlxVlmImeError("画像の PNG エンコードに失敗しました")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Geometry helpers (ported from ollama_vlm_ime.py)
# ---------------------------------------------------------------------------

def detect_patient_record_panel3(frame: np.ndarray) -> Optional[tuple[int, int]]:
    """患者記録画面の第3ペインの x 座標範囲を検出する。"""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=200,
        minLineLength=h // 3,
        maxLineGap=30,
    )
    if lines is None:
        return None

    vert_xs: list[int] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle > 75:
            vert_xs.append(int(x1))

    if not vert_xs:
        return None

    vert_xs.sort()
    clusters: list[int] = []
    current: list[int] = [vert_xs[0]]
    for x in vert_xs[1:]:
        if x - current[-1] < 30:
            current.append(x)
        else:
            clusters.append(int(np.mean(current)))
            current = [x]
    clusters.append(int(np.mean(current)))

    main_dividers = [x for x in clusters if 50 < x < w - 50]
    if len(main_dividers) < 3:
        return None

    return (main_dividers[1], main_dividers[2])


def crop_to_input_region(frame: np.ndarray) -> np.ndarray:
    """フレームをテキスト入力領域にクロップする（第3ペイン検出 → フォールバック全画面）。"""
    panel = detect_patient_record_panel3(frame)
    if panel is not None:
        x1, x2 = panel
        return frame[:, x1:x2]
    return frame


def _crop_center_band(
    frame: np.ndarray,
    top_ratio: float = 0.25,
    bottom_ratio: float = 0.75,
) -> np.ndarray:
    h = frame.shape[0]
    return frame[int(h * top_ratio):int(h * bottom_ratio), :]


def _find_dark_region_y(frame: np.ndarray) -> Optional[int]:
    """IME 反転表示（暗い背景または Windows 10 青色選択バー）の y 座標を検出する。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 閾値 150: 白背景(>200)を除く暗め・中間色（青選択バーも捕捉）
    _, dark_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fh = frame.shape[0]
    best = None
    best_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 20 or w > 800 or h < 12 or h > 100:
            continue
        # タスクバー除外（下15%）
        if y > fh * 0.85:
            continue
        area = w * h
        if area > best_area:
            best_area = area
            best = (y, h)

    if best is None:
        return None
    y, h = best
    return y + h // 2


def _crop_popup_region(frame: np.ndarray) -> np.ndarray:
    """IME ポップアップ候補リストの領域を切り出す。"""
    center_y = _find_dark_region_y(frame)
    h = frame.shape[0]

    if center_y is not None:
        # 選択中候補の上に他候補が表示されるため、上方向に十分なマージンを取る
        y1 = max(0, center_y - 120)
        y2 = min(h, center_y + 300)
        return frame[y1:y2, :]

    return _crop_center_band(frame, top_ratio=0.08, bottom_ratio=0.85)


# ---------------------------------------------------------------------------
# mlx_vlm API call
# ---------------------------------------------------------------------------

def _call_mlx_vlm_with_image(
    image_data_url: str,
    prompt: str,
    *,
    model: str = MLX_VLM_IME_MODEL,
    url: str = MLX_VLM_IME_URL,
    timeout: float = MLX_VLM_IME_TIMEOUT,
) -> str:
    """mlx_vlm.server の OpenAI 互換エンドポイントに画像付きリクエストを送信する。"""
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "stream": False,
        "max_tokens": 128,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
    except (TimeoutError, socket.timeout) as exc:
        raise MlxVlmImeError(
            f"mlx_vlm IME リクエストが {timeout:g} 秒でタイムアウトしました"
        ) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise MlxVlmImeError(
                f"mlx_vlm IME リクエストが {timeout:g} 秒でタイムアウトしました"
            ) from exc
        raise MlxVlmImeError(
            f"mlx_vlm IME への接続に失敗しました: {reason}. endpoint={url}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise MlxVlmImeError("mlx_vlm IME 応答の JSON を解析できませんでした") from exc

    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise MlxVlmImeError(
            f"mlx_vlm IME 応答に content テキストがありません: {result!r}"
        ) from exc

    if not isinstance(content, str):
        raise MlxVlmImeError(f"mlx_vlm IME 応答に content テキストがありません: {result!r}")
    return content.strip()


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _parse_candidate_response(content: str) -> Optional[str]:
    """VLM 応答から単一候補文字列を抽出する。"""
    # <think>...</think> ブロックを除去（Qwen3 の思考出力）
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            candidate = payload.get("candidate")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
            if candidate is None:
                return None
    sq = re.search(r"'candidate'\s*:\s*'([^']+)'", content)
    if sq:
        return sq.group(1).strip() or None
    return None


def _parse_candidates_response(content: str) -> list[str]:
    """VLM 応答から候補リストを抽出する（ポップアップ用）。"""
    # <think>...</think> ブロックを除去
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    cleaned = re.sub(r"```[a-z]*\n?", "", content).strip().rstrip("`").strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            candidates = parsed.get("candidates", [])
            if isinstance(candidates, list):
                result = []
                for c in candidates:
                    c = re.sub(r'^[0-9]+[\s.\-、。]*', '', str(c).strip()).strip()
                    if c:
                        result.append(c)
                return result
        except (json.JSONDecodeError, AttributeError):
            pass
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_inline_candidate_roi(roi: np.ndarray) -> Optional[str]:
    """インライン変換の ROI 画像から現在の変換候補を読み取る。

    Args:
        roi: IME 反転ブロックの ROI 画像

    Returns:
        読み取った候補文字列、または None
    """
    prompt = (
        "この画像は Windows IME の変換候補ウィンドウです。"
        "黒く反転（ハイライト）されている行の文字列だけを正確に読み取ってください。"
        "前後に余計な句読点や記号を付け足さないでください。"
        "読めない場合は null を返してください。"
        "JSONのみで回答してください（余分なコメントや思考過程は不要）。"
        ' 形式: {"candidate": "候補文字列"} または {"candidate": null}'
    )
    data_url = _encode_image_data_url(roi)
    content = _call_mlx_vlm_with_image(data_url, prompt)
    return _parse_candidate_response(content)


def read_inline_candidate_context(frame: np.ndarray) -> Optional[str]:
    """全画面フレームから入力フィールド行の変換候補を読み取る（フォールバック用）。

    Args:
        frame: 全画面フレーム

    Returns:
        変換中の文字列のみ、または None
    """
    cropped = _crop_center_band(crop_to_input_region(frame), top_ratio=0.05, bottom_ratio=0.75)
    prompt = (
        "この画像はWindowsの画面の一部（入力フィールド周辺）です。"
        "テキスト入力フィールドの中で、下線または黒背景・白文字で強調表示されている、"
        "現在変換中の文字列（1〜4文字程度の漢字やひらがな）だけを読み取ってください。"
        "前後に入力済みの文字列は含めないでください。変換中の部分だけを返してください。"
        "見つかった場合はその文字列のみを返してください。見つからない場合は null を返してください。"
        "JSONのみで回答してください（余分なコメントや思考過程は不要）。"
        ' 形式: {"candidate": "文字列"} または {"candidate": null}'
    )
    data_url = _encode_image_data_url(cropped)
    content = _call_mlx_vlm_with_image(data_url, prompt)
    return _parse_candidate_response(content)


def read_popup_candidates(frame: np.ndarray) -> list[str]:
    """全画面フレームから IME ポップアップ候補リスト全体を読み取る。

    Args:
        frame: 全画面フレーム（ポップアップ表示中）

    Returns:
        候補文字列のリスト（表示順）。失敗時は空リスト。
    """
    cropped = _crop_popup_region(crop_to_input_region(frame))
    prompt = (
        "この画像には Windows 日本語IME の変換候補リストが表示されています。"
        "候補リスト（縦に並んだ変換候補のポップアップ）を探し、"
        "1番目（ハイライトまたは選択中の行）から上から下の順に、全ての候補を配列で返してください。"
        "各候補の先頭の数字（1, 2, 3...）は除いて、テキストだけを返してください。"
        "候補リスト以外のテキスト（入力中の文章や画面上の他のテキスト）は含めないでください。"
        "JSONのみで回答してください（余分なコメントや思考過程は不要）。"
        ' 形式: {"candidates": ["候補1", "候補2", "候補3", ...]}'
    )
    data_url = _encode_image_data_url(cropped)
    try:
        content = _call_mlx_vlm_with_image(data_url, prompt)
        return _parse_candidates_response(content)
    except MlxVlmImeError as exc:
        print(f"  [mlx_vlm IME候補リスト] 取得失敗: {exc}")
        return []
