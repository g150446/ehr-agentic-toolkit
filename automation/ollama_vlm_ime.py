"""Ollama vision API helpers for IME candidate reading.

Uses qwen3.5:9b (or configurable model) via Ollama /api/generate with image support.
Images are cropped to the relevant region before sending to reduce processing time.
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

OLLAMA_VLM_IME_URL = os.getenv(
    "OLLAMA_VLM_IME_URL",
    "http://localhost:11434/api/generate",
)
OLLAMA_VLM_IME_MODEL = os.getenv("OLLAMA_VLM_IME_MODEL", "qwen3.5:9b")
OLLAMA_VLM_IME_TIMEOUT = float(os.getenv("OLLAMA_VLM_IME_TIMEOUT", "90"))


class OllamaVlmImeError(RuntimeError):
    """Raised when Ollama VLM IME call fails."""


_MIN_FRAME_HEIGHT = 80
_MIN_FRAME_WIDTH = 200


def _ensure_min_size(frame: np.ndarray) -> np.ndarray:
    """Ollama VLM が処理できる最小サイズに拡大する。

    qwen3.5:9b は非常に小さい画像（高さ < 80px 等）でモデルランナーがクラッシュする。
    最低でも (_MIN_FRAME_HEIGHT, _MIN_FRAME_WIDTH) になるようアスペクト比を保ちながら拡大する。
    """
    h, w = frame.shape[:2]
    scale = max(_MIN_FRAME_HEIGHT / h, _MIN_FRAME_WIDTH / w, 1.0)
    if scale <= 1.0:
        return frame
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _frame_to_b64(frame: np.ndarray) -> str:
    """Encode a numpy BGR frame as base64 JPEG string (no data URI prefix).

    Small frames are upscaled to avoid Ollama model runner crashes.
    """
    frame = _ensure_min_size(frame)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise OllamaVlmImeError("画像の JPEG エンコードに失敗しました")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def detect_patient_record_panel3(
    frame: np.ndarray,
) -> Optional[tuple[int, int]]:
    """患者記録画面の第3ペインの x 座標範囲を検出する。

    患者記録画面は3本の垂直な仕切り線によって4つの領域に分割される。
    左から3番目の領域（仕切り線[1]〜仕切り線[2]）がカルテ入力フィールドを含む。

    Args:
        frame: 全画面フレーム (BGR)

    Returns:
        (x1, x2): 第3ペインの左端・右端 x 座標。検出失敗時は None。
    """
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

    # Cluster nearby x positions into single divider locations
    clusters: list[int] = []
    current: list[int] = [vert_xs[0]]
    for x in vert_xs[1:]:
        if x - current[-1] < 30:
            current.append(x)
        else:
            clusters.append(int(np.mean(current)))
            current = [x]
    clusters.append(int(np.mean(current)))

    # Remove screen edges (within 50px of left/right border)
    main_dividers = [x for x in clusters if 50 < x < w - 50]

    # Need at least 3 dividers to form 4 panels
    if len(main_dividers) < 3:
        return None

    x1 = main_dividers[1]
    x2 = main_dividers[2]
    return (x1, x2)


def crop_to_input_region(frame: np.ndarray) -> np.ndarray:
    """フレームをテキスト入力領域にクロップする。

    患者記録画面が検出された場合は第3ペインに絞る。
    それ以外は全画面をそのまま返す。

    Args:
        frame: 全画面フレーム (BGR)

    Returns:
        テキスト入力領域の画像
    """
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
    """画面の中央帯を切り出す（タイトルバー・タスクバーを除去）。

    Args:
        frame: 全画面フレーム
        top_ratio: 上端から除去する割合（0.25 = 上25%を除去）
        bottom_ratio: 上端からの終端割合（0.75 = 下25%を除去）

    Returns:
        切り出した中央帯の画像
    """
    h = frame.shape[0]
    y1 = int(h * top_ratio)
    y2 = int(h * bottom_ratio)
    return frame[y1:y2, :]


def _find_dark_region_y(frame: np.ndarray) -> Optional[int]:
    """フレーム内で IME 反転表示（暗い背景）の y 座標を検出する。

    IME インライン/ポップアップの選択行は暗い背景（黒 or 紺）で表示される。
    この関数は最も大きな暗い矩形領域の中心 y 座標を返す。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 20 or w > 800 or h < 12 or h > 100:
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
    """ポップアップ候補リストの領域を切り出す。

    IME ポップアップは反転行の近くに縦並びで表示される（最大9候補 × 約20px = 約180px）。
    反転行を検出してその周囲を拡張した帯を切り出す。
    検出失敗時は画面中央帯にフォールバックする。
    """
    center_y = _find_dark_region_y(frame)
    h = frame.shape[0]

    if center_y is not None:
        y1 = max(0, center_y - 30)
        y2 = min(h, center_y + 220)
        return frame[y1:y2, :]

    # フォールバック: 画面中央帯
    return _crop_center_band(frame, top_ratio=0.20, bottom_ratio=0.80)


def call_ollama_vlm(
    image_b64: str,
    prompt: str,
    *,
    model: str = OLLAMA_VLM_IME_MODEL,
    url: str = OLLAMA_VLM_IME_URL,
    timeout: float = OLLAMA_VLM_IME_TIMEOUT,
) -> str:
    """Ollama vision API を呼び出してテキスト応答を返す。

    Args:
        image_b64: base64 エンコードされた画像（data URI プレフィックスなし）
        prompt: プロンプト文字列
        model: Ollama モデル名
        url: Ollama API エンドポイント
        timeout: タイムアウト秒数

    Returns:
        モデルのテキスト応答

    Raises:
        OllamaVlmImeError: 接続失敗・タイムアウト・応答エラー時
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
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
        raise OllamaVlmImeError(
            f"Ollama VLM リクエストが {timeout:g} 秒でタイムアウトしました"
        ) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise OllamaVlmImeError(
                f"Ollama VLM リクエストが {timeout:g} 秒でタイムアウトしました"
            ) from exc
        raise OllamaVlmImeError(
            f"Ollama VLM への接続に失敗しました: {reason}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise OllamaVlmImeError("Ollama VLM 応答の JSON を解析できませんでした") from exc

    response = result.get("response")
    if not isinstance(response, str):
        raise OllamaVlmImeError(f"Ollama VLM 応答に response フィールドがありません: {result!r}")
    return response.strip()


def _parse_candidate_response(content: str) -> Optional[str]:
    """VLM 応答から候補文字列を抽出する。"""
    # JSON オブジェクト {"candidate": "..."} を探す
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
    # シングルクォート形式 {'candidate': '...'} を試みる
    sq = re.search(r"'candidate'\s*:\s*'([^']+)'", content)
    if sq:
        return sq.group(1).strip() or None
    return None


def _parse_candidates_response(content: str) -> list[str]:
    """VLM 応答から候補リストを抽出する（ポップアップ用）。"""
    # コードブロック除去
    cleaned = re.sub(r"```[a-z]*\n?", "", content).strip().rstrip("`").strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            candidates = parsed.get("candidates", [])
            if isinstance(candidates, list):
                result = []
                for c in candidates:
                    c = str(c).strip()
                    # 先頭の数字を除去（例: "2 淫蕩" → "淫蕩"）
                    c = re.sub(r'^[0-9]+[\s.\-、。]*', '', c).strip()
                    if c:
                        result.append(c)
                return result
        except (json.JSONDecodeError, AttributeError):
            pass
    return []


def read_inline_candidate_roi(roi: np.ndarray) -> Optional[str]:
    """インライン変換の ROI 画像から現在の変換候補を読み取る。

    ROI は _find_ime_candidate_region() で既にトリミング済みの小さい画像。

    Args:
        roi: IME 反転ブロックの ROI 画像（色反転済み）

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
    b64 = _frame_to_b64(roi)
    content = call_ollama_vlm(b64, prompt)
    return _parse_candidate_response(content)


def read_inline_candidate_context(frame: np.ndarray) -> Optional[str]:
    """全画面フレームから入力フィールド行の変換候補を読み取る（フォールバック用）。

    フレームを中央帯にクロップしてから Ollama に送信する。
    インライン変換では入力文字列の一部が白黒反転表示される。
    変換中の部分（1〜4文字程度）のみを返す。

    Args:
        frame: 全画面フレーム

    Returns:
        変換中の文字列のみ、または None
    """
    cropped = _crop_center_band(crop_to_input_region(frame), top_ratio=0.25, bottom_ratio=0.75)
    prompt = (
        "この画像はWindowsの画面の一部（入力フィールド周辺）です。"
        "テキスト入力フィールドの中で、下線または黒背景・白文字で強調表示されている、"
        "現在変換中の文字列（1〜4文字程度の漢字やひらがな）だけを読み取ってください。"
        "前後に入力済みの文字列は含めないでください。変換中の部分だけを返してください。"
        "見つかった場合はその文字列のみを返してください。見つからない場合は null を返してください。"
        "JSONのみで回答してください（余分なコメントや思考過程は不要）。"
        ' 形式: {"candidate": "文字列"} または {"candidate": null}'
    )
    b64 = _frame_to_b64(cropped)
    content = call_ollama_vlm(b64, prompt)
    return _parse_candidate_response(content)


def read_popup_candidates(frame: np.ndarray) -> list[str]:
    """全画面フレームから IME ポップアップ候補リスト全体を読み取る。

    ポップアップ領域を検出してクロップしてから Ollama に送信する。

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
    b64 = _frame_to_b64(cropped)
    try:
        content = call_ollama_vlm(b64, prompt)
        return _parse_candidates_response(content)
    except OllamaVlmImeError as exc:
        print(f"  [Ollama VLM候補リスト] 取得失敗: {exc}")
        return []
